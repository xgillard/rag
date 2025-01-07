"""Script qui permet d'ouvrir un tunnel ssh (forward) vers un hote ssh."""

import os
import select
import socketserver
from asyncio import Transport

import paramiko
import paramiko.auth_strategy
import paramiko.client


class ForwardServer(socketserver.ThreadingTCPServer):
    """Un server tcp simple qui gère les requetes entrantes via des threads."""

    daemon_threads = True
    allow_reuse_address = True


class TunnelHandler(socketserver.BaseRequestHandler):
    """Un handler qui redirige le traffic d'une requete (tcp) vers un channel ssh."""

    ssh_transport: paramiko.Transport
    chain_host: str
    chain_port: int

    def handle(self) -> None:
        """Effectue la redirection."""
        try:
            chan = self.ssh_transport.open_channel(
                "direct-tcpip",
                (self.chain_host, self.chain_port),
                self.request.getpeername(),
            )

            if chan is None:
                raise ValueError("channel creation rejected")  # noqa: EM101, TRY003

            while True:
                r, w, x = select.select([self.request, chan], [], [])
                if self.request in r:
                    data = self.request.recv(1024)
                    if len(data) == 0:
                        break
                    chan.send(data)
                if chan in r:
                    data = chan.recv(1024)
                    if len(data) == 0:
                        break
                    self.request.send(data)
        finally:
            chan.close()
            self.request.close()


def forward_tunnel(
    transport: Transport,
    local_port: int,
    remote_host: str,
    remote_port: int,
) -> None:
    """Ouvre un tunnel ssh (forward).

    Ouvre le port local  `local_port` et redirige tout le traffic
    entrant sur ce port vers `remote_host:remote_port` en routant
    tout le traffic sur un canal ssh.
    """

    class SubHandler(TunnelHandler):
        ssh_transport = transport
        chain_host = remote_host
        chain_port = remote_port

    ForwardServer(("", local_port), SubHandler).serve_forever()


if __name__ == "__main__":
    # recupération des secrets.
    HOST = os.environ["SSH_TUNNEL_HOST"]
    UNAME= os.environ["SSH_USER"]
    UPASS= os.environ["SSH_PASS"]
    LOCAL_PORT = os.environ["SSH_TUNNEL_LOCAL_PORT"]
    REMOTE_PORT = os.environ["SSH_TUNNEL_REMOTE_PORT"]
    #
    print(f"uname {UNAME}")
    print(f"upass {UPASS}")
    print(f"host  {HOST}")
    # setup du client.
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())  # noqa: S507
    client.connect(hostname=HOST, username=UNAME, password=UPASS)
    # établissement effectif du tunnel.
    forward_tunnel(client.get_transport(),
                   remote_host=HOST,
                   local_port=int(LOCAL_PORT),
                   remote_port=int(REMOTE_PORT),
                   )
    # cloture de la session
    client.close()