




# from flask import Flask, render_template, jsonify, request
# import cv2
# import numpy as np
# import base64
# from ultralytics import YOLO
# import google.generativeai as genai
# import requests
# import time
# from threading import Lock

# app = Flask(__name__)
# app.secret_key = "visionassist2025"

# # Configure Gemini API (use your key)
# GEMINI_API_KEY = "AIzaSyDl9ZLcFVhC956XjWpGQ74MamMsCxbwalA"
# genai.configure(api_key=GEMINI_API_KEY)
# gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# # Load YOLO model
# yolo_model = YOLO("yolov8n.pt")

# # Store states
# latest_objects = []
# camera_active = False
# speaking_paused = False
# camera_lock = Lock()

# def detect_objects(frame):
#     """YOLO detection function (from your previous working code)"""
#     results = yolo_model(frame, conf=0.20, verbose=False)
#     detected = []
    
#     for r in results:
#         for box in r.boxes:
#             cls_id = int(box.cls[0])
#             label = yolo_model.names[cls_id]
#             detected.append(label)
    
#     # Remove duplicates
#     return list(dict.fromkeys(detected))

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/api/analyze-frame', methods=['POST'])
# def analyze_frame():
#     """Working detection from your previous code"""
#     global latest_objects
    
#     try:
#         data = request.get_json()
#         image_data = data.get('image_data', '')
        
#         # Decode image
#         if ',' in image_data:
#             image_data = image_data.split(',')[1]
        
#         image_bytes = base64.b64decode(image_data)
#         nparr = np.frombuffer(image_bytes, np.uint8)
#         frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
#         # YOLO detection (using your function)
#         detected_objects = detect_objects(frame)
        
#         # Generate description
#         if detected_objects:
#             objects_text = ", ".join(detected_objects)
#             message = f"I can see {objects_text}"
            
#             # Get Gemini description
#             try:
#                 prompt = f"Describe this scene briefly for a blind person: {objects_text}. Give a natural, helpful description in one sentence."
#                 response = gemini_model.generate_content(prompt)
#                 description = response.text
#             except:
#                 description = message
#         else:
#             description = "I do not see any objects in front of you."
#             message = "No objects detected"
        
#         # Store latest
#         with camera_lock:
#             latest_objects = detected_objects
        
#         return jsonify({
#             "status": "success",
#             "detectedObjects": detected_objects,
#             "audioDescription": description,
#             "simpleMessage": message
#         })
        
#     except Exception as e:
#         return jsonify({"status": "error", "error": str(e)}), 500

# @app.route('/api/camera-control', methods=['POST'])
# def camera_control():
#     """Control camera state"""
#     global camera_active, speaking_paused
    
#     data = request.get_json()
#     action = data.get('action', '')
    
#     if action == 'start':
#         camera_active = True
#         speaking_paused = False
#         return jsonify({"status": "success", "message": "Camera started"})
#     elif action == 'stop':
#         camera_active = False
#         return jsonify({"status": "success", "message": "Camera stopped"})
#     elif action == 'pause':
#         speaking_paused = True
#         return jsonify({"status": "success", "message": "Voice paused"})
#     elif action == 'resume':
#         speaking_paused = False
#         return jsonify({"status": "success", "message": "Voice resumed"})
    
#     return jsonify({"status": "error"}), 400

# @app.route('/api/geocode', methods=['POST'])
# def geocode():
#     """Convert address to coordinates using Nominatim (FREE, no API key)"""
#     try:
#         data = request.get_json()
#         address = data.get('address', '')
        
# #         if not address:
# #             return jsonify({"error": "No address"}), 400
        
# #         # Use OpenStreetMap Nominatim (FREE)
# #         url = "https://nominatim.openstreetmap.org/search"
# #         params = {
# #             'q': address,
# #             'format': 'json',
# #             'limit': 1
# #         }
        
# #         headers = {
# #             'User-Agent': 'VisionAssist/1.0'  # Required by Nominatim
# #         }
        
# #         response = requests.get(url, params=params, headers=headers)
# #         data = response.json()
        
# #         if data and len(data) > 0:
# #             return jsonify({
# #                 "lat": float(data[0]['lat']),
# #                 "lon": float(data[0]['lon']),
# #                 "display_name": data[0]['display_name']
# #             })
# #         else:
# #             return jsonify({"error": "Location not found"}), 404
            
# #     except Exception as e:
# #         return jsonify({"error": str(e)}), 500

# # @app.route('/api/reverse-geocode', methods=['POST'])
# # def reverse_geocode():
# #     """Get address from coordinates"""
# #     try:
# #         data = request.get_json()
# #         lat = data.get('lat')
# #         lon = data.get('lon')
        
# #         url = "https://nominatim.openstreetmap.org/reverse"
# #         params = {
# #             'lat': lat,
# #             'lon': lon,
# #             'format': 'json'
# #         }
        
# #         headers = {'User-Agent': 'VisionAssist/1.0'}
# #         response = requests.get(url, params=params, headers=headers)
# #         data = response.json()
        
# #         return jsonify({"address": data.get('display_name', 'Unknown location')})
        
# #     except Exception as e:
# #         return jsonify({"error": str(e)}), 500

# # if __name__ == '__main__':
# #     print("🌟 VisionAssist AI - WORKING DETECTION + ACCURATE MAPS")
# #     print("📡 Server: http://localhost:5000")
# #     app.run(debug=True, host='0.0.0.0', port=5000)

















# from flask import Flask, render_template, jsonify, request
# import cv2
# import numpy as np
# import base64
# from ultralytics import YOLO
# import google.generativeai as genai
# import requests
# import time
# import pytesseract
# from PIL import Image
# import io
# import os
# from threading import Lock

# app = Flask(__name__)
# app.secret_key = "visionassist2025"

# # ==================== CONFIGURE API KEYS (ADD YOURS HERE) ====================
# # Get your API key from: https://makersuite.google.com/app/apikey
# GEMINI_API_KEY = "AIzaSyDl9ZLcFVhC956XjWpGQ74MamMsCxbwalA"  # Your Gemini API key

# # Configure Tesseract path (change this to your installation path)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# # Configure Gemini
# if GEMINI_API_KEY:
#     genai.configure(api_key=GEMINI_API_KEY)
#     gemini_model = genai.GenerativeModel('gemini-1.5-flash')
# else:
#     gemini_model = None
#     print("⚠️ WARNING: Gemini API key not set. Text explanation will not work.")

# # Load YOLO model
# try:
#     yolo_model = YOLO("yolov8n.pt")
#     print("✅ YOLO model loaded successfully")
# except Exception as e:
#     print(f"❌ Error loading YOLO: {e}")
#     yolo_model = None

# # Store states
# latest_objects = []
# camera_active = False
# speaking_paused = False
# camera_lock = Lock()

# def detect_objects(frame):
#     """YOLO detection function"""
#     if yolo_model is None:
#         return []
    
#     try:
#         results = yolo_model(frame, conf=0.20, verbose=False)
#         detected = []
        
#         for r in results:
#             for box in r.boxes:
#                 cls_id = int(box.cls[0])
#                 label = yolo_model.names[cls_id]
#                 detected.append(label)
        
#         return list(dict.fromkeys(detected))
#     except Exception as e:
#         print(f"Detection error: {e}")
#         return []

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/api/analyze-frame', methods=['POST'])
# def analyze_frame():
#     """Analyze frame with YOLO"""
#     global latest_objects
    
#     try:
#         data = request.get_json()
#         image_data = data.get('image_data', '')
        
#         if ',' in image_data:
#             image_data = image_data.split(',')[1]
        
#         image_bytes = base64.b64decode(image_data)
#         nparr = np.frombuffer(image_bytes, np.uint8)
#         frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
#         # Detect objects
#         detected_objects = detect_objects(frame)
        
#         # Generate description
#         if detected_objects:
#             objects_text = ", ".join(detected_objects)
#             message = f"I can see {objects_text}"
            
#             # Get Gemini description if available
#             description = message
#             if gemini_model:
#                 try:
#                     prompt = f"Describe this scene briefly for a blind person: {objects_text}. Give a natural, helpful description in one sentence."
#                     response = gemini_model.generate_content(prompt)
#                     description = response.text
#                 except:
#                     description = message
#         else:
#             description = "I do not see any objects in front of you."
#             message = "No objects detected"
        
#         with camera_lock:
#             latest_objects = detected_objects
        
#         return jsonify({
#             "status": "success",
#             "detectedObjects": detected_objects,
#             "audioDescription": description,
#             "simpleMessage": message
#         })
        
#     except Exception as e:
#         return jsonify({"status": "error", "error": str(e)}), 500

# @app.route('/api/extract-text', methods=['POST'])
# def extract_text():
#     """Extract text from uploaded document"""
#     try:
#         if 'file' not in request.files:
#             return jsonify({"error": "No file uploaded"}), 400
        
#         file = request.files['file']
        
#         # Read image
#         image_bytes = file.read()
#         image = Image.open(io.BytesIO(image_bytes))
        
#         # Extract text using Tesseract
#         extracted_text = pytesseract.image_to_string(image)
        
#         if not extracted_text.strip():
#             extracted_text = "No text could be extracted from the image."
        
#         return jsonify({
#             "status": "success",
#             "text": extracted_text.strip()
#         })
        
#     except Exception as e:
#         return jsonify({"status": "error", "error": str(e)}), 500

# @app.route('/api/explain-text', methods=['POST'])
# def explain_text():
#     """Explain text using Gemini"""
#     try:
#         data = request.get_json()
#         text = data.get('text', '')
#         prompt_type = data.get('prompt', 'simple')
        
#         if not gemini_model:
#             return jsonify({"error": "Gemini API not configured"}), 500
        
#         if prompt_type == 'simple':
#             prompt = f"Explain this text in simple words for a blind person: {text}"
#         elif prompt_type == 'summary':
#             prompt = f"Give a brief summary of this text: {text}"
#         else:
#             prompt = f"Explain this: {text}"
        
#         response = gemini_model.generate_content(prompt)
        
#         return jsonify({
#             "status": "success",
#             "explanation": response.text
#         })
        
#     except Exception as e:
#         return jsonify({"status": "error", "error": str(e)}), 500

# @app.route('/api/camera-control', methods=['POST'])
# def camera_control():
#     """Control camera state"""
#     global camera_active, speaking_paused
    
#     data = request.get_json()
#     action = data.get('action', '')
    
#     if action == 'start':
#         camera_active = True
#         speaking_paused = False
#         return jsonify({"status": "success", "message": "Camera started"})
#     elif action == 'stop':
#         camera_active = False
#         return jsonify({"status": "success", "message": "Camera stopped"})
#     elif action == 'pause':
#         speaking_paused = True
#         return jsonify({"status": "success", "message": "Voice paused"})
#     elif action == 'resume':
#         speaking_paused = False
#         return jsonify({"status": "success", "message": "Voice resumed"})
    
#     return jsonify({"status": "error"}), 400

# @app.route('/api/geocode', methods=['POST'])
# def geocode():
#     """Convert address to coordinates"""
#     try:
#         data = request.get_json()
#         address = data.get('address', '')
        
#         if not address:
#             return jsonify({"error": "No address"}), 400
        
#         url = "https://nominatim.openstreetmap.org/search"
#         params = {
#             'q': address,
#             'format': 'json',
#             'limit': 1
#         }
        
#         headers = {'User-Agent': 'VisionAssist/1.0'}
#         response = requests.get(url, params=params, headers=headers)
#         data = response.json()
        
#         if data and len(data) > 0:
#             return jsonify({
#                 "lat": float(data[0]['lat']),
#                 "lon": float(data[0]['lon']),
#                 "display_name": data[0]['display_name']
#             })
#         else:
#             return jsonify({"error": "Location not found"}), 404
            
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# @app.route('/api/reverse-geocode', methods=['POST'])
# def reverse_geocode():
#     """Get address from coordinates"""
#     try:
#         data = request.get_json()
#         lat = data.get('lat')
#         lon = data.get('lon')
        
#         url = "https://nominatim.openstreetmap.org/reverse"
#         params = {
#             'lat': lat,
#             'lon': lon,
#             'format': 'json'
#         }
        
#         headers = {'User-Agent': 'VisionAssist/1.0'}
#         response = requests.get(url, params=params, headers=headers)
#         data = response.json()
        
#         return jsonify({"address": data.get('display_name', 'Unknown location')})
        
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     print("="*50)
#     print("🌟 VisionAssist AI Starting...")
#     print("="*50)
#     print("\n✅ Features Active:")
#     print("  • YOLO Object Detection")
#     print("  • GPS Navigation")
#     print("  • Document Text Extraction")
#     print("  • Voice Assistant")
    
#     if gemini_model:
#         print("  • Gemini AI Explanation (ACTIVE)")
#     else:
#         print("  ❌ Gemini AI (Not configured - Add API key)")
    
#     print(f"\n📡 Server: http://localhost:5000")
#     print("="*50)
#     app.run(debug=True, host='0.0.0.0', port=5000)








from flask import Flask, render_template, jsonify, request
import cv2
import numpy as np
import base64
from ultralytics import YOLO
import requests
import pytesseract
from PIL import Image
import io
import os
from threading import Lock

app = Flask(__name__)
app.secret_key = "visionassist2025"

# ==================== CONFIGURE GROQ API KEY ====================
# Get your free API key from: https://console.groq.com/keys
GROQ_API_KEY = "gsk_Your_Groq_API_Key_Heregsk_Your_Groq_API_Key_Heregsk_crbVP9LdgZ8mEHTI7YKOWGdyb3FYy92Of5sHze9JlTiCbLiHpr8v"  # Replace with your Groq API key
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

# Configure Tesseract path (change if needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load YOLO model
try:
    yolo_model = YOLO("yolov8n.pt")
    print("✅ YOLO model loaded")
except Exception as e:
    print(f"❌ YOLO error: {e}")
    yolo_model = None

# Store states
latest_objects = []
camera_active = False
speaking_paused = False
camera_lock = Lock()

def ask_groq(prompt, system_message="You are a helpful assistant for blind people."):
    """Ask Groq API for explanation"""
    if not GROQ_API_KEY or GROQ_API_KEY == "gsk_Your_Groq_API_Key_Here":
        return "Please add your Groq API key in app.py"
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "mixtral-8x7b-32768",  # Fast model
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }
    
    try:
        response = requests.post(GROQ_API_URL, json=data, headers=headers)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"Groq API error: {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

def detect_objects(frame):
    """YOLO detection"""
    if yolo_model is None:
        return []
    
    try:
        results = yolo_model(frame, conf=0.20, verbose=False)
        detected = []
        
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = yolo_model.names[cls_id]
                detected.append(label)
        
        return list(dict.fromkeys(detected))
    except:
        return []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/analyze-frame', methods=['POST'])
def analyze_frame():
    """Analyze frame with YOLO"""
    global latest_objects
    
    try:
        data = request.get_json()
        image_data = data.get('image_data', '')
        
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        detected_objects = detect_objects(frame)
        
        if detected_objects:
            objects_text = ", ".join(detected_objects)
            message = f"I can see {objects_text}"
            
            # Get Groq description
            prompt = f"A blind person can see: {objects_text}. Describe this scene briefly and helpfully in 1 sentence."
            description = ask_groq(prompt, "You are an AI helping blind people navigate.")
        else:
            description = "No objects detected in front of you."
            message = "No objects detected"
        
        with camera_lock:
            latest_objects = detected_objects
        
        return jsonify({
            "status": "success",
            "detectedObjects": detected_objects,
            "audioDescription": description,
            "simpleMessage": message
        })
        
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/api/extract-text', methods=['POST'])
def extract_text():
    """Extract text from document"""
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file"}), 400
        
        file = request.files['file']
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Extract text
        extracted_text = pytesseract.image_to_string(image)
        
        if not extracted_text.strip():
            extracted_text = "No text could be extracted."
        
        return jsonify({
            "status": "success",
            "text": extracted_text.strip()
        })
        
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/api/explain-text', methods=['POST'])
def explain_text():
    """Explain text using Groq"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        prompt_type = data.get('prompt', 'simple')
        
        if prompt_type == 'simple':
            prompt = f"Explain this text in very simple words for a blind person: {text}"
        elif prompt_type == 'summary':
            prompt = f"Give a brief summary of this text: {text}"
        else:
            prompt = f"Explain this: {text}"
        
        explanation = ask_groq(prompt, "You are helping a blind person understand text from documents.")
        
        return jsonify({
            "status": "success",
            "explanation": explanation
        })
        
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/api/voice-ask', methods=['POST'])
def voice_ask():
    """Voice assistant using Groq"""
    try:
        data = request.get_json()
        question = data.get('question', '')
        context = data.get('context', '')
        
        if context:
            prompt = f"Context: {context}\n\nQuestion from blind person: {question}\n\nAnswer helpfully and briefly:"
        else:
            prompt = f"Question from blind person: {question}\n\nAnswer helpfully and briefly:"
        
        answer = ask_groq(prompt, "You are a voice assistant for blind people. Give clear, concise answers.")
        
        return jsonify({
            "status": "success",
            "answer": answer
        })
        
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500

@app.route('/api/camera-control', methods=['POST'])
def camera_control():
    """Control camera state"""
    global camera_active, speaking_paused
    
    data = request.get_json()
    action = data.get('action', '')
    
    if action == 'start':
        camera_active = True
        speaking_paused = False
        return jsonify({"status": "success"})
    elif action == 'stop':
        camera_active = False
        return jsonify({"status": "success"})
    elif action == 'pause':
        speaking_paused = True
        return jsonify({"status": "success"})
    elif action == 'resume':
        speaking_paused = False
        return jsonify({"status": "success"})
    
    return jsonify({"status": "error"}), 400

@app.route('/api/geocode', methods=['POST'])
def geocode():
    """Convert address to coordinates"""
    try:
        data = request.get_json()
        address = data.get('address', '')
        
        url = "https://nominatim.openstreetmap.org/search"
        params = {'q': address, 'format': 'json', 'limit': 1}
        headers = {'User-Agent': 'VisionAssist/1.0'}
        
        response = requests.get(url, params=params, headers=headers)
        data = response.json()
        
        if data:
            return jsonify({
                "lat": float(data[0]['lat']),
                "lon": float(data[0]['lon']),
                "display_name": data[0]['display_name']
            })
        else:
            return jsonify({"error": "Location not found"}), 404
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/reverse-geocode', methods=['POST'])
def reverse_geocode():
    """Get address from coordinates"""
    try:
        data = request.get_json()
        lat = data.get('lat')
        lon = data.get('lon')
        
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {'lat': lat, 'lon': lon, 'format': 'json'}
        headers = {'User-Agent': 'VisionAssist/1.0'}
        
        response = requests.get(url, params=params, headers=headers)
        data = response.json()
        
        return jsonify({"address": data.get('display_name', 'Unknown')})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("="*60)
    print("🌟 VISIONASSIST AI WITH GROQ API")
    print("="*60)
    
    if GROQ_API_KEY and GROQ_API_KEY != "gsk_Your_Groq_API_Key_Heregsk_crbVP9LdgZ8mEHTI7YKOWGdyb3FYy92Of5sHze9JlTiCbLiHpr8v":
        print("✅ Groq API: CONFIGURED")
    else:
        print("⚠️  Groq API: NOT CONFIGURED - Add your key in app.py")
        print("   Get free key from: https://console.groq.com/keys")
    
    print("\n📡 Server: http://localhost:5000")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)