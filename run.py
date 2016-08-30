# Really just a lightly modified version of https://github.com/ramhiser/serverless-cloud-vision

import urllib2
import base64
import os
import json

from googleapiclient import discovery
from oauth2client.client import GoogleCredentials

credentials_path = 'credentials.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path

API_URL = 'https://{api}.googleapis.com/$discovery/rest?version={apiVersion}'
DETECT_TYPES = ("LABEL_DETECTION", "TEXT_DETECTION", "SAFE_SEARCH_DETECTION",
                "FACE_DETECTION", "LANDMARK_DETECTION", "LOGO_DETECTION",
                "IMAGE_PROPERTIES")

def get_vision_service():
    credentials = GoogleCredentials.get_application_default()
    return discovery.build('vision', 'v1',
                           credentials=credentials,
                           discoveryServiceUrl=API_URL)

def detect_image(image_url, detect_type="FACE_DETECTION", max_results=4):
    """Uses Google Cloud Vision API to detect an entity within an image in the
    given URL.
    Args:
        image_url: a URL containing an image
        detect_type: detection type to perform (default: facial detection)
        max_results: the maximum number of entities to detect within the image
    Returns:
        An array of dicts with information about the entities detected within
        the image.
    """
    if detect_type not in DETECT_TYPES:
        raise TypeError('Invalid detection type given')

    if type(max_results) is not int or max_results <= 0:
        raise TypeError('Maximum results must be a positive integer')

    img = urllib2.urlopen(image_url)
    if img.headers.maintype != 'image':
        raise TypeError('Invalid filetype given')

    batch_request = [{
        'image': {
            'content': base64.b64encode(img.read()).decode('UTF-8')
        },
        'features': [{
            'type': detect_type,
            'maxResults': max_results
        }]
    }]

    img.close()

    service = get_vision_service()
    request = service.images().annotate(
        body={
            'requests': batch_request
        }
    )
    response = request.execute()

    return response

for img in [
            'https://storage.googleapis.com/fridge-vision.appspot.com/fridge-3.jpg',
            'https://storage.googleapis.com/fridge-vision.appspot.com/fridge-2.jpg',
            'https://storage.googleapis.com/fridge-vision.appspot.com/fridge-1.jpg',
            ]:
    print json.dumps(detect_image(img, detect_type='LABEL_DETECTION', max_results=10), indent=2)
