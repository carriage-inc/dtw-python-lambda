import json
import base64
import mfcc_dtw

def calc_mfcc_dtw(event, context):
    payload = json.loads(event['body'])
    model_binary = base64.b64decode(payload['model'])
    shadow_binary = base64.b64decode(payload['shadow'])
    script_shadow_binary = base64.b64decode(payload['script_shadow'])

    with open('/tmp/model.webm', 'wb') as f:
        f.write(model_binary)
    
    with open('/tmp/shadow.webm', 'wb') as f:
        f.write(shadow_binary)

    with open('/tmp/script_shadow.webm', 'wb') as f:
        f.write(script_shadow_binary)

    model_binary = mfcc_dtw.get_mfcc('/tmp/model.webm')
    shadow_mfcc = mfcc_dtw.get_mfcc('/tmp/shadow.webm')
    script_shadow_binary = mfcc_dtw.get_mfcc('/tmp/script_shadow.webm')

    result = mfcc_dtw.get_with_mfcc(model_binary, shadow_mfcc, script_shadow_binary)
    

    response = {
        "statusCode": 200,
        "headers": {
            "Access-Control-Allow-Origin" : "*",
            "Access-Control-Allow-Credentials": "true"
        },
        "body": json.dumps(result)
    }

    return response
