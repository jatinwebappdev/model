import requests
import json
import base64
# host = "https://p7nshmpes8xh8n-8000.proxy.runpod.net"
#host = "http://38.147.83.29:8000"
host = "http://127.0.0.1:8000"
image = open("miimagen.jpg", "rb").read()
print("image",image)
#Outputs are saved in output directory
def upscale_vary(image, params: dict) -> dict:
    """
    Upscale or Vary
    """
    params["input_image"] = base64.b64encode(image).decode('utf-8') 
    response = requests.post(url=f"{host}/v2/generation/image-upscale-vary",
                        data=json.dumps(params),
                        headers={"Content-Type": "application/json"},
                        timeout=300)
    return response.json()

result =upscale_vary(image=image,
                     params={
                         "uov_method": "Vary (Subtle)",
                         "require_base64": True,
                         "async_process": False
                     })
print(json.dumps(result, indent=4, ensure_ascii=False))
