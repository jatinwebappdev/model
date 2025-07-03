"""Generate API V2 routes

"""
from typing import List
from fastapi import APIRouter, Depends, Header, Query, UploadFile

from fooocusapi.models.common.base import EnhanceCtrlNets, GenerateMaskRequest, DescribeImageType
from fooocusapi.utils.api_utils import api_key_auth
from fooocusapi.models.requests_v1 import ImagePrompt
from fooocusapi.models.requests_v2 import (
    ImageEnhanceRequestJson, ImgInpaintOrOutpaintRequestJson,
    ImgPromptRequestJson,
    Text2ImgRequestWithPrompt,
    ImgUpscaleOrVaryRequestJson
)
from fooocusapi.models.common.response import (
    AsyncJobResponse,
    GeneratedImageResult,
    StopResponse,
    DescribeImageResponse
)
from fooocusapi.utils.call_worker import (
    call_worker,
    generate_mask as gm
)
from fooocusapi.utils.img_utils import base64_to_stream
from fooocusapi.configs.default import img_generate_responses
from fooocusapi.worker import process_stop

from repositories.Fooocus.modules.util import HWC3
from fooocusapi.utils.img_utils import read_input_image
secure_router = APIRouter(
    dependencies=[Depends(api_key_auth)]
)

def stop_worker():
    """Interrupt worker process"""
    process_stop()

def describe_image(
    image: UploadFile,
    image_type: DescribeImageType = Query(
        DescribeImageType.photo,
        description="Image type, 'Photo' or 'Anime'")):
    """\nDescribe image\n
    Describe image, Get tags from an image
    Arguments:
        image {UploadFile} -- Image to get tags
        image_type {DescribeImageType} -- Image type, 'Photo' or 'Anime'
    Returns:
        DescribeImageResponse -- Describe image response, a string
    """
   
    from extras.wd14tagger import default_interrogator as default_interrogator_anime
    interrogator = default_interrogator_anime
    img = HWC3(read_input_image(image))
    result = interrogator(img)
    return result

@secure_router.post(
        path="/v2/generation/image-upscale-vary",
        response_model=List[GeneratedImageResult] | AsyncJobResponse,
        responses=img_generate_responses,
        tags=["GenerateV2"])
def img_upscale_or_vary(
    req: ImgUpscaleOrVaryRequestJson,
    accept: str = Header(None),
    accept_query: str | None = Query(
        None, alias='accept', description="Parameter to override 'Accept' header, 'image/png' for output bytes")):
    """\nImage upscale or vary\n
    Image upscale or vary
    Arguments:
        req {ImgUpscaleOrVaryRequestJson} -- Image upscale or vary request
        accept {str} -- Accept header
        accept_query {str} -- Parameter to override 'Accept' header, 'image/png' for output bytes
    Returns:
            Response -- img_generate_responses    
    """
    if accept_query is not None and len(accept_query) > 0:
        accept ="image/png"
    prompt = describe_image(req.input_image)
    req.input_image = base64_to_stream(req.input_image)

    default_image_prompt = ImagePrompt(cn_img=None)
    image_prompts_files: List[ImagePrompt] = []
    for image_prompt in req.image_prompts:
        image_prompt.cn_img = base64_to_stream(image_prompt.cn_img)
        image = ImagePrompt(
            cn_img=image_prompt.cn_img,
            cn_stop=image_prompt.cn_stop,
            cn_weight=image_prompt.cn_weight,
            cn_type=image_prompt.cn_type)
        image_prompts_files.append(image)
    while len(image_prompts_files) <= 4:
        image_prompts_files.append(default_image_prompt)
    req.image_prompts = image_prompts_files
    print(prompt)
    return call_worker(req, accept, prompt)


@secure_router.post(
        path="v2/generation/stop",
        response_model=StopResponse,
         description="Job stopping",
        tags=["Default"])
def stop():
    "stop/interrupt worker process"
    stop_worker()
    return StopResponse(msg='success')
