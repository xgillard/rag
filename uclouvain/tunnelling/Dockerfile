# The tunnelling container is only meant to spawn an ssh tunnel whenever you
# need a placeholder server that simply relays to an other server that has 
# actually been deployed.
#
# All you need to do is to specify the following secrets:
# - SSH_USER
# - SSH_PASS
# - SSH_TUNNEL_HOST
# - SSH_TUNNEL_LOCAL_PORT
# - SSH_TUNNEL_REMOTE_PORT
FROM python:3.12


COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY ./src/forward_tunnel.py forward_tunnel.py

# Launch postgrest at startup.
CMD [ "python", "./forward_tunnel.py" ]