"""Private OpenAI-compatible gateway settings; never copy credentials into configs."""
import json
import os
from pathlib import Path
from urllib.parse import urlsplit,urlunsplit


def normalize_base_url(base):
    u=urlsplit(str(base).strip())
    if u.scheme!='https' or not u.hostname or u.username or u.password or u.query or u.fragment:
        raise ValueError('Gateway must be an HTTPS base URL without embedded credentials/query')
    path=u.path.rstrip('/') or '/v1'
    return urlunsplit((u.scheme,u.netloc,path,'',''))


def load_gateway():
    path=Path(os.environ.get('GRAPHSKILLEVO_GATEWAY_FILE',str(Path.home()/'.config/graphskillevo/gateway.json')))
    data=json.loads(path.read_text()) if path.is_file() else {}
    base=os.environ.get('GATEWAY_BASE_URL') or data.get('base_url')
    if not base:return None
    key=os.environ.get('GATEWAY_API_KEY') or data.get('api_key')
    if not isinstance(key,str) or not key.strip():raise ValueError('Gateway API key missing in private configuration')
    model=os.environ.get('GATEWAY_MODEL_NAME') or os.environ.get('MODEL_NAME') or data.get('model')
    if not isinstance(model,str) or not model.strip():raise ValueError('Gateway model name missing')
    return dict(base_url=normalize_base_url(base),api_key=key.strip(),model=model.strip())
