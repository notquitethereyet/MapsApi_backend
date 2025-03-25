from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta
import time
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import tempfile
import logging
from werkzeug.utils import secure_filename
from flask_socketio import SocketIO, emit
import json
from datetime import timezone

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
socketio = SocketIO(app, cors_allowed_origins="*")

# Configuration
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Google Maps API key (should be stored as an environment variable in production)
API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY")

# Log API key status (without revealing the key)
if API_KEY:
    logger.info("Google Maps API key found in environment variables")
else:
    logger.warning("Google Maps API key not found in environment variables. API calls will fail.")
    # For development only, you can set a default key - remove in production
    API_KEY = "your_api_key_here"

# Base URL for Google Distance Matrix API
BASE_URL = "https://maps.googleapis.com/maps/api/distancematrix/json"

def allowed_file(filename):
    """Check if the file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def standard_response(data, status=200, message=None):
    """Standardized API response format"""
    response = {
        "data": data,
        "status": status,
        "message": message
    }
    return jsonify(response), status

def calculate_unix_time(departure_date, departure_time):
    """
    Calculate the UNIX timestamp for a specific departure date and time.
    
    Args:
        departure_date (str): The date in "YYYY-MM-DD" format.
        departure_time (str): The time in "HH:MM:SS" (24-hour) format.
        
    Returns:
        int: The UNIX timestamp.
    """
    try:
        # Combine the date and time into a single datetime object
        dt = datetime.strptime(f"{departure_date} {departure_time}", "%Y-%m-%d %H:%M:%S")
        
        # Convert to UNIX timestamp
        unix_timestamp = int(time.mktime(dt.timetuple()))
        return unix_timestamp
    except ValueError as e:
        logger.error(f"Error parsing date/time: {e}")
        return None

def fetch_distance(origin, destination, origin_code, destination_code, origin_address, 
                   destination_address, mode, dest_mode, departure_time):
    """
    Fetch the distance and ETA from the Google Distance Matrix API.
    Special cases:
    - If either source or destination mode is "TELE", the ETA is 0.
    - If source is TRANSIT and destination is DRIVE, or vice versa, handle appropriate modes.
    """
    # Handle telephonic mode for source or destination
    if mode.lower() == "tele" or dest_mode.lower() == "tele":
        return {
            "origin_code": origin_code,
            "origin_address": origin_address,
            "destination_code": destination_code,
            "destination_address": destination_address,
            "transport_mode": mode if mode.lower() == "tele" else dest_mode,
            "timeInMinutes": 0,
        }

    # Determine mode parameter for origin
    api_mode = "driving" if mode.lower() == "drive" else "transit"

    params = {
        "origins": origin,
        "destinations": destination,
        "key": API_KEY,
        "mode": api_mode,
        "departure_time": departure_time,
        "traffic_model": "pessimistic" if api_mode == "driving" else None,
    }

    # Log the API request parameters, especially the departure time
    logger.info(f"Google Maps API Request - Origin: {origin}, Destination: {destination}")
    logger.info(f"Google Maps API Request - Mode: {api_mode}, Departure Time (Unix): {departure_time}")
    
    # Convert Unix timestamp to human-readable format for logging
    try:
        readable_time = datetime.utcfromtimestamp(int(departure_time)).strftime('%Y-%m-%d %H:%M:%S UTC')
        logger.info(f"Google Maps API Request - Departure Time (Human Readable): {readable_time}")
        
        # Check if the time seems off by 7 hours (common UTC/PDT confusion)
        # This is a safeguard against incorrect time conversion
        hour = int(readable_time.split(' ')[1].split(':')[0])
        if hour > 20:  # If the hour is very late, it might be incorrect
            logger.warning(f"Departure time appears to be offset - please verify: {readable_time}")
    except (ValueError, TypeError) as e:
        logger.warning(f"Could not convert departure time to readable format: {e}")
    
    try:
        response = requests.get(BASE_URL, params=params)
        data = response.json()

        if data.get("status") == "OK":
            element = data["rows"][0]["elements"][0]
            if element.get("status") == "OK":
                duration = element["duration"]["value"] / 60  # Convert to minutes
                return {
                    "origin_code": origin_code,
                    "origin_address": origin_address,
                    "destination_code": destination_code,
                    "destination_address": destination_address,
                    "transport_mode": mode,
                    "timeInMinutes": round(duration, 2),
                }
            else:
                logger.error(f"Element error for origin: {origin_code}, destination: {destination_code}: {element['status']}")
        else:
            logger.error(f"API error for origin: {origin_code}, destination: {destination_code}: {data.get('status')}")
    except Exception as e:
        logger.error(f"Exception for origin: {origin_code}, destination: {destination_code}: {e}")
    return None

@app.route('/api/convert-time', methods=['POST'])
def convert_time():
    """
    Convert a local time to UTC using TimeAPI.io
    """
    try:
        data = request.get_json()
        local_time = data.get('localTime')
        
        if not local_time:
            return standard_response(
                None,
                status=400,
                message="Missing localTime parameter"
            )
            
        logger.info(f"Received local time: {local_time}")
        
        # Call TimeAPI.io to convert the time
        timeapi_url = "https://timeapi.io/api/Conversion/ConvertTimeZone"
        payload = {
            "fromTimeZone": "America/Los_Angeles",
            "dateTime": local_time,
            "toTimeZone": "UTC",
            "dstAmbiguity": ""
        }
        
        logger.info(f"Calling TimeAPI with payload: {payload}")
        
        response = requests.post(timeapi_url, json=payload)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        timeapi_data = response.json()
        logger.info(f"TimeAPI response: {timeapi_data}")
        
        # Extract the converted time
        conversion_result = timeapi_data.get('conversionResult', {})
        utc_time = conversion_result.get('dateTime')
        dst_active = conversion_result.get('dstActive', False)
        
        # Calculate Unix timestamp from the UTC time returned by TimeAPI
        utc_dt = datetime.strptime(utc_time, "%Y-%m-%dT%H:%M:%S")
        utc_dt = utc_dt.replace(tzinfo=timezone.utc)  # Explicitly set timezone to UTC
        unix_timestamp = int(utc_dt.timestamp())
        
        # Debug: Convert Unix timestamp back to UTC time to verify
        debug_utc_time = datetime.utcfromtimestamp(unix_timestamp).strftime('%Y-%m-%dT%H:%M:%S')
        logger.info(f"Debug - Unix timestamp: {unix_timestamp}, converted back to UTC: {debug_utc_time}")
        
        # Log the conversion
        logger.info(f"Time conversion: {local_time} -> {utc_time}Z (DST is {'active' if dst_active else 'not active'})")
        
        return standard_response(
            {
                "utcTime": utc_time,
                "unixTimestamp": unix_timestamp,
                "dstActive": dst_active
            },
            status=200,
            message="Time converted successfully"
        )
    
    except Exception as e:
        logger.error(f"Error in convert_time: {str(e)}")
        return standard_response(
            None,
            status=500,
            message=f"Server error: {str(e)}"
        )

@app.route('/api/upload-distance-matrix', methods=['POST'])
def upload_distance_matrix():
    """
    Process an uploaded Excel file for distance matrix calculation.
    """
    try:
        # Check if the post request has the file part
        if 'file' not in request.files:
            return standard_response(
                None, 
                status=400, 
                message="No file part in the request"
            )
            
        file = request.files['file']
        departure_date = request.form.get('departureDate')
        departure_time = request.form.get('departureTime')
        timestamp = request.form.get('timestamp')  # Changed from 'unixTimestamp' to match frontend
        
        # Log all form data for debugging
        logger.info(f"Form data received: date={departure_date}, time={departure_time}, timestamp={timestamp}")
        
        # Validate departure date and time
        if not departure_date or not departure_time:
            return standard_response(
                None,
                status=400,
                message="Missing departure date or time"
            )
            
        # If user does not select file, browser might submit an empty part without filename
        if file.filename == '':
            return standard_response(
                None, 
                status=400, 
                message="No selected file"
            )
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            logger.info(f"File uploaded: {filename}")
            
            # Use provided Unix timestamp if available, otherwise calculate it
            if timestamp:
                unix_time = int(timestamp)
                logger.info(f"Using provided Unix timestamp: {unix_time}")
            else:
                # Parse the departure date and time to get UNIX timestamp
                unix_time = calculate_unix_time(departure_date, departure_time)
                logger.info(f"Calculated Unix timestamp: {unix_time} for date: {departure_date}, time: {departure_time}")
            
            if not unix_time:
                return standard_response(
                    None,
                    status=400,
                    message="Invalid departure date or time format"
                )
            
            # Process the file based on its type
            if filename.endswith('.csv'):
                addresses_df = pd.read_csv(filepath)
            else:  # xlsx or xls
                addresses_df = pd.read_excel(filepath)
            
            # Validate required columns
            required_columns = ['Address Code', 'Address', 'Transport Mode']
            missing_columns = [col for col in required_columns if col not in addresses_df.columns]
            if missing_columns:
                return standard_response(
                    None,
                    status=400,
                    message=f"Missing required columns: {', '.join(missing_columns)}"
                )
            
            # Extract relevant columns
            address_codes = addresses_df['Address Code']
            addresses = addresses_df['Address']
            transport_modes = addresses_df['Transport Mode']
            
            # Prepare tasks for all address pairs
            tasks = []
            for i, origin in enumerate(addresses):
                origin_code = address_codes.iloc[i]
                origin_address = addresses.iloc[i]
                origin_mode = transport_modes.iloc[i]
                for j, destination in enumerate(addresses):
                    if i != j:  # Skip same-origin and destination pairs
                        destination_code = address_codes.iloc[j]
                        destination_address = addresses.iloc[j]
                        destination_mode = transport_modes.iloc[j]
                        tasks.append((
                            origin, destination, origin_code, destination_code, 
                            origin_address, destination_address, origin_mode, 
                            destination_mode, unix_time
                        ))
            
            # Generate a unique job ID for tracking progress
            job_id = f"job_{int(time.time())}"
            total_tasks = len(tasks)
            completed_tasks = 0
            
            # Emit initial progress
            socketio.emit('progress_update', {
                'job_id': job_id,
                'completed': completed_tasks,
                'total': total_tasks,
                'percent': 0,
                'status': 'processing'
            })
            
            # Use ThreadPoolExecutor for multithreading (limit number of workers to avoid API rate limits)
            results = []
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_task = {
                    executor.submit(fetch_distance, *task): task for task in tasks
                }
                
                for future in as_completed(future_to_task):
                    result = future.result()
                    if result:
                        results.append(result)
                    
                    # Update progress
                    completed_tasks += 1
                    percent_complete = int((completed_tasks / total_tasks) * 100)
                    
                    # Emit progress update every 5% or for every 10 tasks
                    if completed_tasks % max(1, total_tasks // 20) == 0 or completed_tasks == total_tasks:
                        socketio.emit('progress_update', {
                            'job_id': job_id,
                            'completed': completed_tasks,
                            'total': total_tasks,
                            'percent': percent_complete,
                            'status': 'processing' if completed_tasks < total_tasks else 'completed'
                        })
                        logger.info(f"Progress: {completed_tasks}/{total_tasks} ({percent_complete}%)")
            
            # Convert results to DataFrame
            results_df = pd.DataFrame(results)
            
            # Save the results to a temporary file
            result_filename = f"distance_matrix_{int(time.time())}.xlsx"
            result_filepath = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
            results_df.to_excel(result_filepath, index=False)
            
            # Clean up the input file
            os.remove(filepath)
            
            # Final progress update
            socketio.emit('progress_update', {
                'job_id': job_id,
                'completed': total_tasks,
                'total': total_tasks,
                'percent': 100,
                'status': 'completed',
                'result_filename': result_filename
            })
            
            # Return all results
            return standard_response(
                {
                    "message": "Distance matrix calculation completed successfully",
                    "totalPairs": len(results),
                    "sampleData": results,
                    "resultFilename": result_filename,
                    "job_id": job_id
                },
                message="Distance matrix calculation completed"
            )
        
        return standard_response(
            None, 
            status=400, 
            message="File type not allowed. Please upload XLSX, XLS, or CSV files only."
        )
        
    except Exception as e:
        logger.error(f"Error processing distance matrix: {str(e)}")
        return standard_response(
            None, 
            status=500, 
            message=f"Error processing distance matrix: {str(e)}"
        )

@app.route('/api/download-result/<filename>', methods=['GET'])
def download_result(filename):
    """
    Download the processed result file.
    """
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(filename))
        if os.path.exists(filepath):
            return send_from_directory(
                app.config['UPLOAD_FOLDER'],
                secure_filename(filename),
                as_attachment=True
            )
        else:
            return standard_response(
                None,
                status=404,
                message="Result file not found"
            )
    except Exception as e:
        logger.error(f"Error downloading result file: {str(e)}")
        return standard_response(
            None,
            status=500,
            message=f"Error downloading result file: {str(e)}"
        )

@app.route('/api/calculate-distance', methods=['POST'])
def calculate_distance():
    """
    Calculate distance between two addresses.
    Expected input: {
        "originAddress": "123 Main St, City, Country",
        "destinationAddress": "456 Elm St, City, Country",
        "transportMode": "drive",
        "departureTime": "2025-03-24T15:30:00.000Z"
    }
    """
    try:
        data = request.json
        origin = data.get('originAddress')
        destination = data.get('destinationAddress')
        mode = data.get('transportMode', 'drive').lower()
        departure_time_str = data.get('departureTime')
        
        if not origin or not destination or not departure_time_str:
            return standard_response(
                None,
                status=400,
                message="Missing required parameters: originAddress, destinationAddress, or departureTime"
            )
        
        # Parse departure time
        departure_time = datetime.fromisoformat(departure_time_str.replace('Z', '+00:00'))
        unix_time = int(departure_time.timestamp())
        
        # Determine mode parameter
        api_mode = "driving" if mode == "drive" else "transit"
        
        params = {
            "origins": origin,
            "destinations": destination,
            "key": API_KEY,
            "mode": api_mode,
            "departure_time": unix_time,
            "traffic_model": "pessimistic" if api_mode == "driving" else None,
        }

        # Log the API request parameters, especially the departure time
        logger.info(f"Google Maps API Request - Origin: {origin}, Destination: {destination}")
        logger.info(f"Google Maps API Request - Mode: {api_mode}, Departure Time (Unix): {unix_time}")
        
        # Convert Unix timestamp to human-readable format for logging
        try:
            readable_time = datetime.utcfromtimestamp(int(unix_time)).strftime('%Y-%m-%d %H:%M:%S UTC')
            logger.info(f"Google Maps API Request - Departure Time (Human Readable): {readable_time}")
            
            # Check if the time seems off by 7 hours (common UTC/PDT confusion)
            # This is a safeguard against incorrect time conversion
            hour = int(readable_time.split(' ')[1].split(':')[0])
            if hour > 20:  # If the hour is very late, it might be incorrect
                logger.warning(f"Departure time appears to be offset - please verify: {readable_time}")
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not convert departure time to readable format: {e}")
        
        response = requests.get(BASE_URL, params=params)
        data = response.json()
        
        if data.get("status") == "OK":
            element = data["rows"][0]["elements"][0]
            if element.get("status") == "OK":
                duration = element["duration"]["value"] / 60  # Convert to minutes
                distance = element["distance"]["value"] / 1000  # Convert to kilometers
                
                result = {
                    "origin": origin,
                    "destination": destination,
                    "transportMode": mode,
                    "timeInMinutes": round(duration, 2),
                    "distanceInKm": round(distance, 2),
                    "departureTime": departure_time_str,
                }
                
                return standard_response(
                    result,
                    message="Distance calculated successfully"
                )
            else:
                return standard_response(
                    None,
                    status=400,
                    message=f"Error calculating distance: {element.get('status')}"
                )
        else:
            return standard_response(
                None,
                status=400,
                message=f"Error from Google Distance Matrix API: {data.get('status')}"
            )
    
    except Exception as e:
        logger.error(f"Error calculating distance: {str(e)}")
        return standard_response(
            None,
            status=500,
            message=f"Error calculating distance: {str(e)}"
        )

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Simple health check endpoint to verify the API is running
    """
    return standard_response(
        {"status": "healthy", "timestamp": datetime.now().isoformat()},
        message="API is running"
    )

@app.route('/health', methods=['GET'])
def root_health_check():
    """
    Root-level health check endpoint for easier testing
    """
    return jsonify({
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "message": "API is running"
    })

if __name__ == '__main__':
    # Import here to avoid circular import
    from flask import send_from_directory
    
    # Check if running in development or production
    if os.environ.get('RAILWAY_ENVIRONMENT') == 'production':
        # In production, gunicorn will serve the app
        # We don't need to run the app here
        pass
    else:
        # In development, use socketio to run the app
        socketio.run(app, debug=True, port=5000, allow_unsafe_werkzeug=True)