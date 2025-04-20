import os
import time
import logging
import json
import tempfile
import math # Needed for ceiling division if using other strategies, but not strictly needed for the current fix. Keep for potential future use.
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry # Corrected import path
from flask import Flask, request, jsonify, send_from_directory, make_response # Added make_response
from flask_cors import CORS
from werkzeug.utils import secure_filename

# --- Configuration ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Ensure CORS allows requests from your React app's origin in production
# For Railway, you might set the allowed origin via an environment variable
allowed_origin = os.environ.get("CORS_ALLOWED_ORIGIN", "*") # Default to wildcard for dev, be specific in prod
CORS(app, resources={r"/api/*": {"origins": allowed_origin}})

UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Google Maps API Key - **NEVER HARDCODE IN PRODUCTION**
API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")
if not API_KEY:
    logger.warning("GOOGLE_MAPS_API_KEY environment variable not set. API calls will likely fail.")
    # You might want to uncomment the line below ONLY for local development,
    # but it's strongly recommended to use environment variables.
    # API_KEY = "YOUR_DEV_API_KEY_HERE"
    # Or raise an error if the key is critical for the app to function:
    # raise ValueError("GOOGLE_MAPS_API_KEY environment variable is required.")
else:
    logger.info("Successfully loaded Google Maps API key from environment variables.")

# Google API Constants
BASE_URL = "https://maps.googleapis.com/maps/api/distancematrix/json"
# --- IMPORTANT API LIMITS for Standard Plan ---
# These MUST respect Google's limits to avoid MAX_DIMENSIONS_EXCEEDED / MAX_ELEMENTS_EXCEEDED
MAX_ORIGINS_PER_BATCH = 10      # Maximum origins per API call (limit is 25) - Choose a lower value for flexibility
MAX_DESTINATIONS_PER_BATCH = 10 # Maximum destinations per API call (limit is 25) - Choose a lower value for flexibility
MAX_ELEMENTS_PER_QUERY = 100    # origins * destinations <= 100 (This MUST hold: MAX_ORIGINS * MAX_DESTINATIONS <= 100)

# Check if chosen batch sizes respect the element limit
if MAX_ORIGINS_PER_BATCH * MAX_DESTINATIONS_PER_BATCH > MAX_ELEMENTS_PER_QUERY:
    raise ValueError(f"Configuration Error: MAX_ORIGINS_PER_BATCH ({MAX_ORIGINS_PER_BATCH}) * "
                     f"MAX_DESTINATIONS_PER_BATCH ({MAX_DESTINATIONS_PER_BATCH}) exceeds "
                     f"MAX_ELEMENTS_PER_QUERY ({MAX_ELEMENTS_PER_QUERY}). Reduce batch sizes.")


# --- Utility Functions ---

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def standard_response(data, status=200, message=None):
    """Standardized API response format"""
    response = {
        "data": data,
        "status": status,
        "message": message if message else ("Success" if 200 <= status < 300 else "Error")
    }
    # Using jsonify automatically sets the Content-Type header to application/json
    return jsonify(response), status


def create_requests_session_with_retries(retries=3, backoff_factor=0.3, status_forcelist=(500, 502, 503, 504)):
    """Creates a requests session with retry logic."""
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def calculate_unix_time(departure_date, departure_time):
    """
    Calculate the UNIX timestamp for a specific departure date and time
    based on the server's local timezone.
    *** Note: This can be unreliable if server timezone differs from expected. ***
    Prefer using timestamps generated client-side via a reliable UTC conversion.

    Args:
        departure_date (str): The date in "YYYY-MM-DD" format.
        departure_time (str): The time in "HH:MM:SS" (24-hour) format.

    Returns:
        int: The UNIX timestamp, or None if parsing fails.
    """
    try:
        dt_naive = datetime.strptime(f"{departure_date} {departure_time}", "%Y-%m-%d %H:%M:%S")
        # Convert using server's local time. Railway servers default to UTC.
        # If your input time assumes a different zone (e.g., PST), this will be wrong.
        unix_timestamp = int(time.mktime(dt_naive.timetuple()))
        # Log the assumed timezone for mktime for debugging
        # Use time.tzset() on Unix-like systems if needed, but generally rely on system config
        try:
            local_tz_name = time.tzname[time.daylight] if time.daylight else time.tzname[0]
        except AttributeError:
             local_tz_name = "Unknown (tzname not available)" # Handle systems where tzname might not work as expected
        logger.warning(f"Calculated Unix timestamp {unix_timestamp} using server's local time ({local_tz_name}) for {departure_date} {departure_time}. This might be incorrect if input time assumed a different zone.")
        return unix_timestamp
    except ValueError as e:
        logger.error(f"Error parsing date/time with strptime/mktime: {e}")
        return None
    except OverflowError as e:
         logger.error(f"Date/time resulted in overflow for mktime: {e}")
         return None

# --- Core Logic for Distance Matrix (Batching) ---

def fetch_distance_batch(session, origin_batch, destination_list, departure_time, api_mode):
    """
    Fetch distances for a batch of origins against a list of destinations using Google API.
    """
    if not API_KEY:
         logger.error("API Key is missing. Cannot make Google Maps API calls.")
         raise ValueError("Google Maps API Key is not configured.")

    if not origin_batch or not destination_list:
        logger.warning("fetch_distance_batch called with empty origin or destination list.")
        return []

    origin_addresses = [o['address'] for o in origin_batch]
    destination_addresses = [d['address'] for d in destination_list]

    # Validate batch sizes against individual limits BEFORE making the call
    if len(origin_addresses) > 25 or len(destination_addresses) > 25:
         logger.error(f"Internal Error: Batch dimensions violate API limits ({len(origin_addresses)} origins, {len(destination_addresses)} destinations). Max 25 each.")
         # Return error markers for all pairs in this invalid batch
         error_results = []
         for o_info in origin_batch:
             for d_info in destination_list:
                 if o_info['code'] != d_info['code']: # Don't create self-errors
                      error_results.append({
                         "error": "INTERNAL_BATCH_LIMIT_ERROR",
                         "origin_code": o_info['code'], "origin_address": o_info['address'],
                         "destination_code": d_info['code'], "destination_address": d_info['address'],
                         "transport_mode": o_info['mode'], "timeInMinutes": None, # Indicate error
                      })
         return error_results


    params = {
        "origins": "|".join(origin_addresses),
        "destinations": "|".join(destination_addresses),
        "key": API_KEY,
        "mode": api_mode,
        "departure_time": departure_time,
        # Only add traffic_model for driving
        **({"traffic_model": "pessimistic"} if api_mode == "driving" else {}),
    }

    results = []
    try:
        # Log essential parts of the request
        try:
             # Use UTC for consistent logging regardless of server time
             readable_time = datetime.fromtimestamp(departure_time, timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')
        except (ValueError, TypeError, OverflowError):
             readable_time = f"Invalid timestamp {departure_time}"
        logger.info(f"Sending batch API request: {len(origin_addresses)} origins vs {len(destination_addresses)} destinations. Mode: {api_mode}. Time: {readable_time} (Unix: {departure_time})")

        response = session.get(BASE_URL, params=params, timeout=60) # Increased timeout slightly
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()

        # Detailed logging of the response status and structure can help debug
        logger.debug(f"API Response Status: {data.get('status')}")
        logger.debug(f"API Response Data (partial): {str(data)[:200]}...") # Log start of response

        if data.get("status") == "OK":
             # Check if rows array exists and matches origin count
             if "rows" not in data or len(data["rows"]) != len(origin_batch):
                 logger.warning(f"API OK status but 'rows' count ({len(data.get('rows', []))}) "
                                f"mismatches origin batch size ({len(origin_batch)}). Response might be incomplete.")
                 # Attempt to process what was received, but be aware of potential issues

             for i, origin_row in enumerate(data.get("rows", [])):
                if i >= len(origin_batch):
                    logger.warning(f"API response row index {i} out of bounds for origin batch size {len(origin_batch)} (processing truncated response).")
                    continue
                origin_info = origin_batch[i]

                # Check if elements array exists and matches destination count
                if "elements" not in origin_row or len(origin_row["elements"]) != len(destination_list):
                     logger.warning(f"API OK status but 'elements' count ({len(origin_row.get('elements', []))}) "
                                    f"mismatches destination list size ({len(destination_list)}) for origin {origin_info['code']}. Response might be incomplete.")
                     # Attempt to process what was received

                for j, element in enumerate(origin_row.get("elements", [])):
                    if j >= len(destination_list):
                        logger.warning(f"API response element index {j} out of bounds for destination list size {len(destination_list)} for origin {origin_info['code']} (processing truncated response).")
                        continue
                    dest_info = destination_list[j]

                    if origin_info['code'] == dest_info['code']: continue # Skip self-pairs

                    # Handle TELE mode - should already be filtered by caller, but acts as a safeguard
                    if origin_info['mode'].lower() == "tele" or dest_info['mode'].lower() == "tele":
                         # This case shouldn't normally be hit if filtering in upload_distance_matrix is correct
                         logger.debug(f"TELE pair found within non-TELE API batch processing: {origin_info['code']} -> {dest_info['code']}")
                         results.append({
                            "origin_code": origin_info['code'], "origin_address": origin_info['address'],
                            "destination_code": dest_info['code'], "destination_address": dest_info['address'],
                            "transport_mode": "tele", "timeInMinutes": 0,
                         })
                         continue

                    # Process API result element
                    element_status = element.get("status")
                    if element_status == "OK":
                        # Prefer duration_in_traffic if available (driving mode with departure time)
                        duration_data = element.get("duration_in_traffic", element.get("duration"))
                        if duration_data and "value" in duration_data:
                            duration_seconds = duration_data["value"]
                            # Handle potential zero duration (e.g., same location, minimal walk for transit)
                            if duration_seconds >= 0:
                                duration_minutes = round(duration_seconds / 60, 2)
                                results.append({
                                    "origin_code": origin_info['code'], "origin_address": origin_info['address'],
                                    "destination_code": dest_info['code'], "destination_address": dest_info['address'],
                                    "transport_mode": origin_info['mode'], # Use the mode of the API call
                                    "timeInMinutes": duration_minutes,
                                })
                            else:
                                logger.warning(f"Negative duration value ({duration_seconds}s) received for {origin_info['code']} -> {dest_info['code']}. Treating as error.")
                                results.append({"error": "INVALID_DURATION_DATA", **origin_info, **dest_info, "timeInMinutes": None })
                        else:
                             logger.warning(f"Missing 'duration' or 'value' in OK element for {origin_info['code']} -> {dest_info['code']}. Element: {element}")
                             results.append({"error": "MISSING_DURATION_DATA", **origin_info, **dest_info, "timeInMinutes": None })
                    # Handle specific non-OK statuses gracefully
                    elif element_status in ["NOT_FOUND", "ZERO_RESULTS"]:
                        logger.warning(f"API Element Status '{element_status}' for {origin_info['code']} -> {dest_info['code']}. Address likely invalid or route impossible.")
                        results.append({"error": element_status, **origin_info, **dest_info, "timeInMinutes": None})
                    else: # Catch-all for other unexpected element statuses
                        logger.error(f"Unhandled API Element Status '{element_status}' for {origin_info['code']} -> {dest_info['code']}. Element: {element}")
                        results.append({"error": element_status or "UNKNOWN_ELEMENT_STATUS", **origin_info, **dest_info, "timeInMinutes": None})

        # Handle Overall API Request Status Errors
        elif data.get("status") == "INVALID_REQUEST":
             logger.error(f"API Error: INVALID_REQUEST. Check parameters (addresses, key, format). Error message: {data.get('error_message', 'N/A')}")
             raise requests.exceptions.RequestException(f"API Error: {data.get('status')} - {data.get('error_message', 'Request format invalid')}") # Signal critical failure
        elif data.get("status") == "MAX_DIMENSIONS_EXCEEDED":
             logger.error(f"API Error: MAX_DIMENSIONS_EXCEEDED ({len(origin_addresses)}x{len(destination_addresses)}). Batching logic failed.")
             raise requests.exceptions.RequestException("API limit exceeded (MAX_DIMENSIONS_EXCEEDED)")
        elif data.get("status") == "MAX_ELEMENTS_EXCEEDED":
             logger.error(f"API Error: MAX_ELEMENTS_EXCEEDED ({len(origin_addresses) * len(destination_addresses)}). Batching logic failed.")
             raise requests.exceptions.RequestException("API limit exceeded (MAX_ELEMENTS_EXCEEDED)")
        elif data.get("status") == "OVER_QUERY_LIMIT":
             logger.error(f"API Error: OVER_QUERY_LIMIT. Check usage limits and billing. Error message: {data.get('error_message', 'N/A')}")
             raise requests.exceptions.RequestException(f"API Error: {data.get('status')} - {data.get('error_message', 'Quota exceeded')}") # Signal critical failure (retry might help temporarily)
        elif data.get("status") == "REQUEST_DENIED":
             logger.error(f"API Error: REQUEST_DENIED. Check API key validity, API enablement, and billing. Error message: {data.get('error_message', 'N/A')}")
             raise requests.exceptions.RequestException(f"API Error: {data.get('status')} - {data.get('error_message', 'Request denied, check key/billing')}") # Signal critical failure
        elif data.get("status") == "UNKNOWN_ERROR":
             logger.error(f"API Error: UNKNOWN_ERROR. Temporary server issue likely. Error message: {data.get('error_message', 'N/A')}")
             raise requests.exceptions.RequestException(f"API Error: {data.get('status')} - {data.get('error_message', 'Unknown server error')}") # Retry might help
        else: # Catch any other unexpected API status
            error_msg = data.get('error_message', 'N/A')
            logger.error(f"Unhandled API Status '{data.get('status')}'. Message: {error_msg}")
            raise requests.exceptions.RequestException(f"Unhandled API Status: {data.get('status')} - {error_msg}")

    except requests.exceptions.Timeout:
        logger.error(f"API request timed out for batch starting with {origin_batch[0]['code']}")
        raise # Re-raise to signal batch failure
    except requests.exceptions.RequestException as e:
        # Includes connection errors, HTTP errors from raise_for_status, and errors raised above
        logger.error(f"API request failed for batch starting with {origin_batch[0]['code']}: {e}")
        raise # Re-raise
    except Exception as e:
        # Catch-all for JSON parsing errors or other unexpected issues
        logger.exception(f"Unexpected error processing batch starting with {origin_batch[0]['code']}: {e}", exc_info=True)
        raise # Re-raise

    return results


# --- Flask Routes ---

@app.route('/api/convert-time', methods=['POST'])
def convert_time():
    """
    (Deprecated in favor of client-side UTC conversion and passing timestamp)
    Convert a local time (assumed to be US/Pacific by default) to UTC using TimeAPI.io
    and return the UTC time string and corresponding Unix timestamp.
    """
    logger.warning("'/api/convert-time' endpoint is deprecated. Client should calculate UTC timestamp directly.")
    try:
        data = request.get_json()
        local_time_str = data.get('localTime') # Expecting "YYYY-MM-DDTHH:MM:SS"
        from_timezone = data.get('fromTimeZone', 'America/Los_Angeles') # Default

        if not local_time_str:
            return standard_response(None, 400, "Missing localTime parameter")

        logger.info(f"Received local time for DEPRECATED conversion: {local_time_str} (From Timezone: {from_timezone})")

        # Parse the ISO format date and convert to the format expected by TimeAPI
        try:
            # Attempt parsing ISO format, allowing for 'Z' or timezone offset
            dt = datetime.fromisoformat(local_time_str.replace('Z', '+00:00'))
            # Format it as expected by TimeAPI: "YYYY-MM-DD HH:MM:SS"
            # Keep it naive for the API call as TimeAPI uses the fromTimeZone parameter
            formatted_time = dt.strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"Formatted time for TimeAPI: {formatted_time}")
        except Exception as e:
            logger.error(f"Error formatting date: {e}. Using original string.")
            formatted_time = local_time_str # Fallback

        timeapi_url = "https://timeapi.io/api/Conversion/ConvertTimeZone"
        payload = {
            "fromTimeZone": from_timezone,
            "dateTime": formatted_time,
            "toTimeZone": "UTC",
            "dstAmbiguity": "" # Let TimeAPI handle ambiguity if any
        }
        logger.info(f"Calling TimeAPI with payload: {json.dumps(payload)}")

        response = requests.post(timeapi_url, json=payload, timeout=10)
        response.raise_for_status() # Check for HTTP errors from TimeAPI
        timeapi_data = response.json()
        logger.info(f"TimeAPI response: {json.dumps(timeapi_data)}")

        # Check for errors within the TimeAPI response structure itself
        if 'conversionResult' not in timeapi_data or not timeapi_data.get('conversionResult'):
            error_details = timeapi_data.get('validationMessages', [])
            logger.error(f"TimeAPI validation error: {error_details}")
            return standard_response(None, 400, f"TimeAPI validation failed: {error_details}")

        conversion_result = timeapi_data.get('conversionResult', {})
        utc_time_str = conversion_result.get('dateTime') # Expected format: "YYYY-MM-DDTHH:MM:SS"
        dst_active = conversion_result.get('dstActive', False)

        if not utc_time_str:
             logger.error(f"TimeAPI did not return a valid dateTime in conversionResult: {conversion_result}")
             return standard_response(None, 500, "Failed to get valid UTC time from TimeAPI.io")

        # Parse the UTC string from TimeAPI to get the timestamp
        try:
             # TimeAPI returns ISO format, so use fromisoformat
             utc_dt = datetime.fromisoformat(utc_time_str)
             # Ensure it's timezone-aware UTC
             utc_dt = utc_dt.replace(tzinfo=timezone.utc)
             unix_timestamp = int(utc_dt.timestamp())
        except ValueError:
             logger.error(f"Could not parse UTC time string from TimeAPI: {utc_time_str}")
             return standard_response(None, 500, "Failed to parse UTC time returned by TimeAPI.io")

        logger.info(f"DEPRECATED Time conversion successful via TimeAPI: {local_time_str} ({from_timezone}) -> {utc_time_str}Z (Unix: {unix_timestamp}).")

        return standard_response(
            {"utcTime": utc_time_str, "unixTimestamp": unix_timestamp, "dstActive": dst_active},
            status=200, message="Time converted successfully using TimeAPI.io (Endpoint Deprecated)"
        )

    except requests.exceptions.RequestException as e:
         logger.error(f"Error calling TimeAPI.io: {str(e)}")
         return standard_response(None, 503, f"Service unavailable: Could not reach TimeAPI.io - {str(e)}")
    except Exception as e:
        logger.exception(f"Error in deprecated convert_time endpoint: {str(e)}", exc_info=True)
        return standard_response(None, 500, f"Internal server error during time conversion: {str(e)}")


@app.route('/api/upload-distance-matrix', methods=['POST'])
def upload_distance_matrix():
    """
    Processes the uploaded file synchronously. Calculates distance matrix
    using batched API calls and returns the filename of the result Excel file.
    """
    start_time = time.time()
    # Use a unique ID for logging and output filename
    request_id = f"req_{int(start_time)}_{os.urandom(4).hex()}"
    logger.info(f"[{request_id}] Starting distance matrix processing...")

    # --- Input Validation ---
    if 'file' not in request.files:
        return standard_response(None, 400, "No file part in the request")

    file = request.files['file']
    # Preferred input: Unix timestamp calculated client-side in UTC
    timestamp_str = request.form.get('timestamp')
    # Fallback (DEPRECATED): Date and Time strings (rely on server time or need zone info)
    departure_date = request.form.get('departureDate')
    departure_time_str = request.form.get('departureTime')


    if file.filename == '':
        return standard_response(None, 400, "No selected file")
    if not allowed_file(file.filename):
        return standard_response(None, 400, f"File type not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

    # --- Determine Departure Timestamp (Prioritize direct timestamp) ---
    unix_departure_time = None
    timestamp_source = "None"
    if timestamp_str:
        try:
            unix_departure_time = int(timestamp_str)
            # Basic sanity check for realistic timestamp range (e.g., not in 1970 or far future)
            if 946684800 < unix_departure_time < 2524608000: # Roughly 2000 to 2050
                 logger.info(f"[{request_id}] Using provided Unix timestamp: {unix_departure_time}")
                 timestamp_source = "Client UTC Timestamp"
            else:
                 logger.warning(f"[{request_id}] Provided timestamp {unix_departure_time} seems out of reasonable range. Ignoring.")
                 unix_departure_time = None # Reset if unreasonable
        except (ValueError, TypeError):
            logger.warning(f"[{request_id}] Invalid timestamp format '{timestamp_str}'. Attempting fallback.")
            unix_departure_time = None

    # Fallback to Date/Time strings (less reliable)
    if unix_departure_time is None and departure_date and departure_time_str:
        logger.warning(f"[{request_id}] Using fallback: calculating Unix timestamp from date/time '{departure_date} {departure_time_str}' using server timezone.")
        unix_departure_time = calculate_unix_time(departure_date, departure_time_str)
        if unix_departure_time is None:
             return standard_response(None, 400, "Invalid date/time format for fallback timestamp calculation.")
        logger.info(f"[{request_id}] Using fallback calculated Unix timestamp: {unix_departure_time}")
        timestamp_source = "Server Time Fallback"
    elif unix_departure_time is None:
         # If no timestamp and no valid date/time fallback
         return standard_response(None, 400, "Missing required departure time information (provide 'timestamp' or 'departureDate'/'departureTime').")

    # --- File Handling & Parsing ---
    filename = secure_filename(file.filename)
    # Use a unique name for the temporary input file to avoid collisions
    temp_input_filename = f"{request_id}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_input_filename)
    df = None
    try:
        file.save(filepath)
        logger.info(f"[{request_id}] File uploaded temporarily to: {filepath}")
        read_start_time = time.time()
        if filename.lower().endswith('.csv'):
            # Try detecting encoding, fall back to utf-8
            try:
                df = pd.read_csv(filepath, encoding='utf-8-sig') # Try utf-8 with BOM first
            except UnicodeDecodeError:
                logger.warning(f"[{request_id}] UTF-8 decoding failed for CSV, trying latin1.")
                df = pd.read_csv(filepath, encoding='latin1')
        else: # xlsx or xls
            df = pd.read_excel(filepath, engine=None) # Let pandas choose engine
        logger.info(f"[{request_id}] File read into DataFrame in {time.time() - read_start_time:.2f}s. Shape: {df.shape}")

    except FileNotFoundError:
         logger.error(f"[{request_id}] File disappeared after saving? Path: {filepath}")
         return standard_response(None, 500, "Internal server error: Uploaded file vanished.")
    except Exception as e:
        logger.error(f"[{request_id}] Error reading or saving file {filepath}: {e}", exc_info=True)
        # Attempt cleanup even if save failed partially
        if os.path.exists(filepath):
            try: os.remove(filepath); logger.info(f"[{request_id}] Cleaned up partially saved/failed input file: {filepath}")
            except Exception as e_rem: logger.error(f"[{request_id}] Error removing temp input file {filepath} after read error: {e_rem}")
        return standard_response(None, 400, f"Error reading uploaded file content: {e}")
    finally:
        # Ensure temporary input file is removed AFTER DataFrame is successfully created (or after error)
         if os.path.exists(filepath):
             try:
                 os.remove(filepath)
                 logger.info(f"[{request_id}] Removed temporary input file: {filepath}")
             except Exception as e_rem:
                 logger.error(f"[{request_id}] Error removing temp input file {filepath}: {e_rem}")

    # --- Data Validation and Preparation ---
    required_columns = ['Address Code', 'Address', 'Transport Mode']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        return standard_response(None, 400, f"Missing required columns in uploaded file: {', '.join(missing_columns)}")

    # Clean and structure data
    locations = []
    invalid_rows = 0
    for index, row in df.iterrows():
        code = str(row['Address Code']).strip()
        address = str(row['Address']).strip()
        mode = str(row['Transport Mode']).strip().lower()

        if not code or not address:
            logger.warning(f"[{request_id}] Skipping row {index+2}: Missing Code or Address.")
            invalid_rows += 1
            continue
        if mode not in ['drive', 'transit', 'tele']:
             logger.warning(f"[{request_id}] Skipping row {index+2} (Code: {code}): Invalid Transport Mode '{mode}'. Must be 'drive', 'transit', or 'tele'.")
             invalid_rows += 1
             continue

        locations.append({'code': code, 'address': address, 'mode': mode})

    logger.info(f"[{request_id}] Prepared {len(locations)} valid locations from input file ({invalid_rows} rows skipped).")
    if not locations:
        return standard_response(None, 400, "No valid locations found in the uploaded file after cleaning.")

    # --- Separate TELE pairs and API-requiring locations ---
    all_results = []
    tele_pairs_generated = set() # Use a set to track unique (origin, destination) pairs for TELE
    api_locations = [] # Locations needing API calls

    for loc in locations:
        if loc['mode'] == 'tele':
            # Generate TELE results for this location paired with ALL OTHER locations
            for other_loc in locations:
                if loc['code'] == other_loc['code']: continue # Skip self-pair

                # Ensure pair uniqueness regardless of direction (A->B is same as B->A for TELE)
                pair_key = tuple(sorted((loc['code'], other_loc['code'])))
                if pair_key not in tele_pairs_generated:
                    # Add A->B direction
                    all_results.append({
                        "origin_code": loc['code'], "origin_address": loc['address'],
                        "destination_code": other_loc['code'], "destination_address": other_loc['address'],
                        "transport_mode": "tele", "timeInMinutes": 0
                    })
                    # Add B->A direction
                    all_results.append({
                        "origin_code": other_loc['code'], "origin_address": other_loc['address'],
                        "destination_code": loc['code'], "destination_address": loc['address'],
                        "transport_mode": "tele", "timeInMinutes": 0
                    })
                    tele_pairs_generated.add(pair_key)
        else: # drive or transit
            api_locations.append(loc)

    logger.info(f"[{request_id}] Pre-calculated {len(all_results)} unique TELE pairs. Processing {len(api_locations)} locations via API.")

    # Filter destinations for API calls - exclude TELE locations
    api_destinations = [loc for loc in api_locations] # Destinations can be drive or transit

    # Separate origins by mode for API calls
    drive_origins = [loc for loc in api_locations if loc['mode'] == 'drive']
    transit_origins = [loc for loc in api_locations if loc['mode'] == 'transit']

    # --- Prepare API Tasks (CORRECTED BATCHING LOGIC) ---
    tasks_to_submit = []
    session = create_requests_session_with_retries()
    api_origins_map = {'driving': drive_origins, 'transit': transit_origins}

    for api_mode, origins_list in api_origins_map.items():
        if not origins_list: continue # Skip if no origins for this mode
        logger.info(f"[{request_id}] Preparing {len(origins_list)} {api_mode} origins...")

        # Iterate through origin batches (respecting MAX_ORIGINS_PER_BATCH)
        for i in range(0, len(origins_list), MAX_ORIGINS_PER_BATCH):
            origin_batch = origins_list[i : i + MAX_ORIGINS_PER_BATCH]

            if not api_destinations: # Skip if no valid destinations
                logger.warning(f"[{request_id}] No valid API destinations found for {api_mode} origin batch starting at index {i}.")
                continue

            # Now, *always* iterate through destination batches (respecting MAX_DESTINATIONS_PER_BATCH)
            for j in range(0, len(api_destinations), MAX_DESTINATIONS_PER_BATCH):
                dest_batch = api_destinations[j : j + MAX_DESTINATIONS_PER_BATCH]

                # Final check: Ensure this specific origin/destination batch combo respects the element limit
                # This check is crucial and uses the actual batch sizes determined by the limits
                if len(origin_batch) * len(dest_batch) <= MAX_ELEMENTS_PER_QUERY:
                    # Ensure batch sizes also respect individual limits (should be guaranteed by loop steps, but double-check)
                     if len(origin_batch) <= MAX_ORIGINS_PER_BATCH and len(dest_batch) <= MAX_DESTINATIONS_PER_BATCH:
                        logger.debug(f"[{request_id}] Adding task: {len(origin_batch)} {api_mode} origins (batch {i//MAX_ORIGINS_PER_BATCH+1}) vs "
                                     f"{len(dest_batch)} destinations (batch {j//MAX_DESTINATIONS_PER_BATCH+1}).")
                        # Pass copies of lists to avoid potential modification issues in threads
                        tasks_to_submit.append((session, list(origin_batch), list(dest_batch), unix_departure_time, api_mode))
                     else:
                         # This case indicates a logic error in how batches were created or limits defined
                         logger.error(f"[{request_id}] INTERNAL LOGIC ERROR: Batch dimensions ({len(origin_batch)}x{len(dest_batch)}) "
                                      f"exceed configured MAX limits ({MAX_ORIGINS_PER_BATCH}x{MAX_DESTINATIONS_PER_BATCH}) "
                                      f"despite passing element check. Skipping sub-batch.")
                else:
                    # This case implies MAX_ORIGINS_PER_BATCH * MAX_DESTINATIONS_PER_BATCH > MAX_ELEMENTS_PER_QUERY
                    # which should have been caught by the initial configuration check.
                    logger.error(f"[{request_id}] CONFIGURATION ERROR: Max batch sizes ({MAX_ORIGINS_PER_BATCH}x{MAX_DESTINATIONS_PER_BATCH}) "
                                 f"inherently exceed MAX_ELEMENTS_PER_QUERY ({MAX_ELEMENTS_PER_QUERY}). Adjust configuration. Skipping sub-batch.")


    # --- Execute Tasks Concurrently ---
    total_api_batches = len(tasks_to_submit)
    completed_api_batches = 0
    total_pairs_processed_api = 0
    api_pair_errors = 0 # Count pairs within batches that had element-level errors
    failed_batch_count = 0 # Count whole batches that failed (e.g., timeout, API key error)
    estimated_pairs_in_failed_batches = 0

    logger.info(f"[{request_id}] Starting {total_api_batches} API batch requests...")

    # Limit workers - important in resource-constrained environments like serverless/small VMs
    # Using os.cpu_count() might give a high number on powerful machines, consider a fixed cap.
    # max_workers = min(10, (os.cpu_count() or 1) + 4) # Heuristic: ~CPU cores + buffer, capped at 10
    max_workers = 10 # Fixed number often works well to avoid overwhelming API or local resources
    logger.info(f"[{request_id}] Using max_workers={max_workers} for ThreadPoolExecutor.")


    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map future back to task arguments for better error logging
        future_to_task = {executor.submit(fetch_distance_batch, *task): task for task in tasks_to_submit}

        # Use tqdm for progress if running interactively, otherwise just log periodically
        # progress_iterator = tqdm(as_completed(future_to_task), total=total_api_batches, desc="Processing API Batches")
        progress_iterator = as_completed(future_to_task)


        for future in progress_iterator:
            task_info = future_to_task[future] # Get original task arguments
            origin_batch, dest_batch = task_info[1], task_info[2] # Extract origin/dest batches for logging
            batch_pair_count = len(origin_batch) * len(dest_batch) # Potential pairs in this batch

            try:
                batch_result = future.result() # result() re-raises exceptions from the worker thread

                if batch_result:
                    # Filter out results explicitly marked with an error during fetch_distance_batch
                    # These are element-level errors (e.g., NOT_FOUND, ZERO_RESULTS)
                    successful_batch_results = []
                    for r in batch_result:
                         if 'error' not in r or r.get("timeInMinutes") is not None: # Keep results even if error marker exists but time was calculated (e.g. for info)
                             successful_batch_results.append(r)
                         else:
                             api_pair_errors += 1 # Count pairs with explicit errors and no time

                    all_results.extend(successful_batch_results)
                    total_pairs_processed_api += len(successful_batch_results)
                    logger.debug(f"[{request_id}] Batch completed successfully. Added {len(successful_batch_results)} results. Encountered {api_pair_errors} pair errors so far.")

            except requests.exceptions.RequestException as exc:
                 # Specific API level errors (Quota, Key, Invalid Request, Max Dimensions etc.) or Network errors
                 logger.error(f'[{request_id}] API Request Exception for batch (origins: {[o["code"] for o in origin_batch]}): {exc}')
                 failed_batch_count += 1
                 estimated_pairs_in_failed_batches += batch_pair_count
            except ValueError as exc: # Catch specific errors like missing API key
                 logger.error(f'[{request_id}] Value Error during batch processing (origins: {[o["code"] for o in origin_batch]}): {exc}')
                 failed_batch_count += 1
                 estimated_pairs_in_failed_batches += batch_pair_count
            except Exception as exc:
                 # Catch-all for unexpected errors within the future execution
                 logger.exception(f'[{request_id}] Unexpected Exception for batch (origins: {[o["code"] for o in origin_batch]}): {exc}', exc_info=True)
                 failed_batch_count += 1
                 estimated_pairs_in_failed_batches += batch_pair_count

            completed_api_batches += 1
            # Log progress periodically
            if total_api_batches > 0 and (completed_api_batches % max(1, total_api_batches // 10) == 0 or completed_api_batches == total_api_batches):
                logger.info(f"[{request_id}] Progress: {completed_api_batches}/{total_api_batches} API batches completed.")


    # --- Finalize and Respond ---
    logger.info(f"[{request_id}] API calls finished.")
    logger.info(f"[{request_id}] Summary: {completed_api_batches}/{total_api_batches} batches processed.")
    if failed_batch_count > 0:
        logger.warning(f"[{request_id}] {failed_batch_count} batches failed entirely (estimated {estimated_pairs_in_failed_batches} pairs missed).")
    if api_pair_errors > 0:
        logger.warning(f"[{request_id}] Encountered {api_pair_errors} individual pair errors within successful batches.")

    logger.info(f"[{request_id}] Total results collected (including TELE): {len(all_results)}")

    # Create DataFrame from results
    results_df = pd.DataFrame(all_results)
    result_filename = None
    final_status = 500 # Default to error unless successful
    message = "An unexpected error occurred during processing."

    if not results_df.empty:
        # Optional: Clean up results - e.g., fill missing times for errored pairs if desired
        # results_df['timeInMinutes'].fillna(-1, inplace=True) # Example: fill errors with -1

        # Define standard column order
        output_columns = ["origin_code", "origin_address", "destination_code", "destination_address", "transport_mode", "timeInMinutes"]
        # Ensure all expected columns exist, add if missing (e.g., if only TELE results)
        for col in output_columns:
            if col not in results_df.columns:
                results_df[col] = None # Or appropriate default like pd.NA

        # Reorder and select columns
        results_df = results_df[output_columns]

        # Sort results for consistent output
        results_df.sort_values(by=["origin_code", "destination_code"], inplace=True)

        # Save to Excel
        result_filename_base = f"distance_matrix_results_{request_id}.xlsx"
        result_filepath = os.path.join(app.config['UPLOAD_FOLDER'], result_filename_base)
        final_pair_count = len(results_df)

        try:
            save_start_time = time.time()
            results_df.to_excel(result_filepath, index=False, engine='openpyxl')
            logger.info(f"[{request_id}] Results saved to {result_filepath} in {time.time() - save_start_time:.2f}s. ({final_pair_count} rows)")
            result_filename = result_filename_base # Set filename only on successful save
            final_status = 200
            message = f"Processing complete. {final_pair_count} pairs calculated (including TELE and potentially errored pairs)."
            if failed_batch_count > 0 or api_pair_errors > 0:
                 error_summary = []
                 if failed_batch_count > 0: error_summary.append(f"{failed_batch_count} batches failed")
                 if api_pair_errors > 0: error_summary.append(f"{api_pair_errors} individual pairs failed")
                 message += f" Warning: {' and '.join(error_summary)}."

        except Exception as e:
             logger.error(f"[{request_id}] Error saving results to Excel file {result_filepath}: {e}", exc_info=True)
             final_status = 500 # Internal error because saving failed
             message = f"Processing generated {final_pair_count} results, but failed to save the Excel file. Error: {e}"
             result_filename = None # Ensure filename is None if save fails
    else:
        # Handle cases where no results were generated at all
        logger.warning(f"[{request_id}] No results (including TELE) were generated after processing.")
        final_status = 200 # Request was valid, just no pairs resulted
        message = "Processing complete, but no valid origin-destination pairs were found or calculable (check input data and API errors)."
        if failed_batch_count > 0 or api_pair_errors > 0:
             message += f" Significant errors occurred during API calls."
             final_status = 400 # If errors likely caused no results, indicate bad request/API issue
        final_pair_count = 0


    processing_time = time.time() - start_time
    logger.info(f"[{request_id}] Request completed in {processing_time:.2f} seconds. Status: {final_status}. Message: {message}")

    return standard_response(
        {
            "message": message, # Keep detailed message here
            "totalPairsInOutput": final_pair_count,
            "resultFilename": result_filename, # Will be null if save failed or no results
            "processingTimeSeconds": round(processing_time, 2),
            "timestampSource": timestamp_source, # Indicate how timestamp was determined
            "failedBatchCount": failed_batch_count,
            "apiPairErrors": api_pair_errors
        },
        status=final_status,
        message=message # Also include main message in standard response wrapper
    )

    # --- Removed broad Exception catch here; specific errors handled above ---
    # --- Error handling within the loop and finalize block should cover most issues ---


# --- Other Routes (Download, Health) ---

@app.route('/api/download-result/<filename>', methods=['GET'])
def download_result(filename):
    """Download the processed result file."""
    logger.info(f"Download request received for: {filename}")

    # Security: Basic check on filename format and prevent directory traversal
    # Ensure it matches the expected pattern and contains no path separators
    expected_prefix = "distance_matrix_results_req_"
    expected_suffix = ".xlsx"
    if '..' in filename or '/' in filename or '\\' in filename or not (
        filename.startswith(expected_prefix) and filename.endswith(expected_suffix)
    ):
         logger.warning(f"Attempt to download invalid or potentially malicious filename: {filename}")
         return standard_response(None, 400, "Invalid filename format.")

    # Use Werkzeug's secure_filename again as a defense-in-depth measure,
    # though the checks above should be sufficient if the pattern is strict.
    safe_filename = secure_filename(filename)
    if safe_filename != filename:
         # This indicates unexpected characters were stripped, which shouldn't happen if pattern matches
         logger.error(f"Filename sanitization changed '{filename}' to '{safe_filename}' unexpectedly. Aborting download.")
         return standard_response(None, 400, "Invalid filename characters detected.")

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)

    if os.path.exists(filepath):
        logger.info(f"Sending file: {filepath} for download.")
        try:
            # Use Flask's send_from_directory for safer file serving
            response = send_from_directory(
                app.config['UPLOAD_FOLDER'],
                safe_filename,
                as_attachment=True,
                # Correct MIME type for modern Excel files
                mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
            # Optionally add headers to prevent caching if files are often regenerated
            # response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            # response.headers["Pragma"] = "no-cache"
            # response.headers["Expires"] = "0"

            # ---- Cleanup Strategy: Decide if/when to delete the file ----
            # Option 1: Delete immediately after sending starts (might fail if download interrupted)
            # Option 2: Schedule cleanup (e.g., cron job, separate task queue) - More robust
            # Option 3: Leave files and let temp system cleanup handle it (simplest, relies on OS)
            # For now, we leave the file (Option 3). Implement cleanup if storage becomes an issue.
            # try:
            #     os.remove(filepath)
            #     logger.info(f"Removed result file after initiating download: {filepath}")
            # except Exception as e_rem:
            #     logger.error(f"Error removing result file {filepath} after download: {e_rem}")
            # ---- End Cleanup Strategy ----

            return response

        except Exception as e_send:
            logger.error(f"Error using send_from_directory for {filepath}: {e_send}", exc_info=True)
            return standard_response(None, 500, "Internal server error while sending file.")
    else:
        logger.warning(f"Result file not found for download: {filepath}")
        return standard_response(None, 404, "Result file not found. It may have expired, failed processing, or been cleaned up.")


@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint for monitoring."""
    api_key_status = "OK" if API_KEY else "MISSING"
    return standard_response(
        {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "apiKeyStatus": api_key_status
        },
        message="API is running"
    )

# Keep the root health check for simple load balancer checks etc.
@app.route('/health', methods=['GET'])
def root_health_check():
    """Root-level health check endpoint for easier testing/LB checks."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }), 200

# Basic root route
@app.route('/', methods=['GET'])
def index():
    """Basic index route indicating the API is up."""
    return jsonify({"message": "Distance Matrix API is running. Use API endpoints."}), 200


# --- App Execution (for Local Dev & Gunicorn/Waitress) ---
if __name__ == '__main__':
    # Ensure UPLOAD_FOLDER exists (especially important for local dev)
    try:
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        logger.info(f"Upload folder set to: {app.config['UPLOAD_FOLDER']}")
    except Exception as e:
         logger.error(f"Could not create upload folder '{app.config['UPLOAD_FOLDER']}': {e}. File uploads will fail.", exc_info=True)
         # Optionally exit if the upload folder is critical
         # exit(1)

    # Determine port from environment variable (common for PaaS like Heroku, Railway)
    port = int(os.environ.get('PORT', 5000)) # Use 5000 to match frontend configuration

    # Use Flask's built-in server for local development (debug=True enables reloader and debugger)
    # **DO NOT USE debug=True IN PRODUCTION**
    # Railway/Heroku/etc. typically use a production WSGI server like Gunicorn or Waitress specified elsewhere (e.g., Procfile)
    # The 'debug' flag controls Flask's built-in debugger and reloader.
    # Set DEBUG environment variable ('true' or '1') for development convenience.
    is_debug_mode = os.environ.get('FLASK_DEBUG', 'false').lower() in ['true', '1']

    logger.info(f"Starting Flask app. Debug mode: {is_debug_mode}. Port: {port}")
    # The host '0.0.0.0' makes the server accessible externally (within its network).
    # Use '127.0.0.1' to only allow connections from the local machine.
    app.run(debug=is_debug_mode, host='0.0.0.0', port=port)

# --- Example Procfile for Railway/Heroku ---
# web: gunicorn --bind 0.0.0.0:$PORT --workers 2 --threads 4 --timeout 120 --log-level info app:app
#
# Explanation:
# - `web:` defines the web process type.
# - `gunicorn`: the WSGI server.
# - `--bind 0.0.0.0:$PORT`: Binds to all interfaces on the port assigned by the platform.
# - `--workers 2`: Number of Gunicorn worker processes (adjust based on CPU/memory). Start low (1-2).
# - `--threads 4`: Number of threads per worker (useful for I/O bound tasks like API calls). Adjust based on testing.
# - `--timeout 120`: Increases request timeout to 120 seconds (Gunicorn default is 30s). Adjust based on expected processing time.
# - `--log-level info`: Gunicorn logging level.
# - `app:app`: Tells Gunicorn the Flask app instance `app` is in the file `app.py`.
# Note: The optimal number of workers/threads depends heavily on the app's nature (CPU vs I/O bound) and the resources of the server/container. `workers * threads` shouldn't exceed available resources significantly.
