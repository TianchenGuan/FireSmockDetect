"""Read runs/<exp>/seg01/status.json from S3 (same ada session as the orchestrator) for the v5b arms."""
import json, sys
from pathlib import Path
sys.path.insert(0, '/local/home/tianrgua/cvproject/cluster')
import orchestrate_training as ot
BUCKET = 'smoke-detect-data-000000000001-use1'
sess, _ = ot.ada_session()
s3 = sess.client('s3', region_name='us-east-1')
for exp in sys.argv[1:]:
    key = f'runs/{exp}/seg01/status.json'
    try:
        st = json.loads(s3.get_object(Bucket=BUCKET, Key=key)['Body'].read())
        print(exp, 'phase=%s' % st.get('phase'), 'host=%s' % st.get('host'), 'ts=%s' % st.get('ts'), 'gpus=%s' % st.get('gpus'), 'run_root=%s' % st.get('run_root'))
    except Exception as e:
        print(exp, 'no status.json yet (%s)' % type(e).__name__)
