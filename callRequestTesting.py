import requests
import json
import base64
import os

# Host options:
# host = "https://p7nshmpes8xh8n-8000.proxy.runpod.net"
# host = "http://38.147.83.29:8000"
host = "http://127.0.0.1:8000"

def upscale_vary(image, params: dict) -> dict:
    """
    Send image to server to upscale or vary using specified parameters.
    """
    # Encode image as base64
    params["input_image"] = base64.b64encode(image).decode('utf-8')
    
    try:
        print(f"Sending request to {host}/v2/generation/image-upscale-vary")
        response = requests.post(
            url=f"{host}/v2/generation/image-upscale-vary",
            data=json.dumps(params),
            headers={"Content-Type": "application/json"},
            timeout=300
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        # If response is not OK, show detailed error info
        if not response.ok:
            print(f"Error response: {response.text}")
            return {
                "error": f"HTTP {response.status_code}", 
                "details": response.text,
                "headers": dict(response.headers)
            }
        
        # Try parsing JSON
        return response.json()
        
    except requests.exceptions.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        print(f"Raw Response Text: {response.text}")
        return {"error": "Invalid JSON response", "content": response.text}
    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}")
        return {"error": "Request failed", "details": str(e)}

def main():
    # Check if image file exists
    image_path = "miimagen.jpg"
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        return
    
    # Load image in binary mode
    try:
        with open(image_path, "rb") as f:
            image = f.read()
        print(f"Successfully loaded image: {len(image)} bytes")
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Test server connectivity first
    try:
        health_response = requests.get(f"{host}/ping", timeout=10)
        print(f"Server health check: {health_response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Server connectivity issue: {e}")
        return
    
    # Call the upscale/vary function
    result = upscale_vary(
        image=image,
        params={
            "uov_method": "Vary (Subtle)",
            "require_base64": True,
            "async_process": False
        }
    )
    
    # Print the result in a readable format
    print("\n--- RESULT ---")
    print(json.dumps(result, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    main()
