#!/usr/bin/env python3
"""
WebSocket signaling proxy for NVIDIA Kit WebRTC streaming.

The Kit streaming SDK (omni.kit.streamsdk.plugins 6.x) generates SDP offers with
c=IN IP4 0.0.0.0 and no ICE candidates. This makes WebRTC unusable from outside
the host because the browser has no routable address to connect to.

This proxy sits between nginx and Kit's signaling port (49100), intercepting
WebSocket messages and rewriting the SDP to use the server's public IP.

Usage:
    python3 signaling_proxy.py --public-ip 52.32.247.131 --listen-port 49200 --kit-port 49100

Architecture:
    Browser → nginx (/streaming/) → THIS PROXY (49200) → Kit signaling (49100)
"""

import asyncio
import json
import logging
import re
import argparse
import signal

try:
    import websockets
    from websockets.server import serve as ws_serve
except ImportError:
    print("ERROR: websockets library required. Install with: pip3 install websockets")
    raise SystemExit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [signaling-proxy] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("signaling-proxy")


def rewrite_sdp(sdp: str, public_ip: str) -> str:
    """Replace 0.0.0.0 and private IPs in SDP with the public IP, and inject ICE candidates."""
    original = sdp
    # Replace 0.0.0.0 in connection lines and RTCP attributes
    sdp = re.sub(r"IN IP4 0\.0\.0\.0", f"IN IP4 {public_ip}", sdp)
    # Replace Docker bridge IPs (172.17.x.x)
    sdp = re.sub(r"IN IP4 172\.17\.\d+\.\d+", f"IN IP4 {public_ip}", sdp)
    # Replace EC2 private IPs (10.x.x.x, 172.31.x.x)
    sdp = re.sub(r"IN IP4 10\.\d+\.\d+\.\d+", f"IN IP4 {public_ip}", sdp)
    sdp = re.sub(r"IN IP4 172\.31\.\d+\.\d+", f"IN IP4 {public_ip}", sdp)
    # Also fix the origin line (o= line uses 127.0.0.1)
    sdp = re.sub(r"IN IP4 127\.0\.0\.1", f"IN IP4 {public_ip}", sdp)

    # Inject ICE host candidates if none exist.
    # Kit's streaming SDK uses ice-lite with trickle but never sends candidates,
    # so the browser has no address to connect to. We inject host candidates
    # for each media section using the port from the m= line.
    if "a=candidate:" not in sdp:
        lines = sdp.split("\r\n")
        new_lines = []
        for line in lines:
            new_lines.append(line)
            # After each m= line, inject a host candidate with the media port
            m = re.match(r"^m=(\w+)\s+(\d+)\s+", line)
            if m:
                media_type = m.group(1)
                port = int(m.group(2))
                if port > 0:
                    # ICE candidate format: foundation component protocol priority ip port typ host
                    # Component 1 = RTP (sufficient with rtcp-mux)
                    # Priority: 2130706431 = host candidate (2^24 * 126 + 2^8 * 65535 + 255)
                    candidate = (
                        f"a=candidate:1 1 udp 2130706431 {public_ip} {port} typ host"
                    )
                    new_lines.append(candidate)
                    log.info("Injected ICE candidate for %s: %s:%s", media_type, public_ip, port)
        sdp = "\r\n".join(new_lines)

    if sdp != original:
        log.info("Rewrote SDP: replaced IPs with %s", public_ip)
    return sdp


def rewrite_ice_candidate(candidate: str, public_ip: str) -> str:
    """Replace IPs in trickle ICE candidate strings."""
    candidate = re.sub(r"0\.0\.0\.0", public_ip, candidate)
    candidate = re.sub(r"172\.17\.\d+\.\d+", public_ip, candidate)
    candidate = re.sub(r"10\.\d+\.\d+\.\d+", public_ip, candidate)
    candidate = re.sub(r"172\.31\.\d+\.\d+", public_ip, candidate)
    return candidate


def rewrite_message(raw: str, public_ip: str) -> str:
    """Inspect a signaling message and rewrite SDP/ICE if present."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw

    if "peer_msg" not in data:
        return raw

    try:
        inner_str = data["peer_msg"]["msg"]
        inner = json.loads(inner_str)
    except (json.JSONDecodeError, TypeError, KeyError):
        return raw

    modified = False

    # Rewrite SDP offer/answer
    if "sdp" in inner:
        new_sdp = rewrite_sdp(inner["sdp"], public_ip)
        if new_sdp != inner["sdp"]:
            inner["sdp"] = new_sdp
            modified = True

    # Rewrite nvstSdp (NVIDIA-specific SDP field)
    if "nvstSdp" in inner:
        new_nvst = rewrite_sdp(inner["nvstSdp"], public_ip)
        if new_nvst != inner["nvstSdp"]:
            inner["nvstSdp"] = new_nvst
            modified = True

    # Rewrite trickle ICE candidates
    if "candidate" in inner and inner["candidate"]:
        new_cand = rewrite_ice_candidate(inner["candidate"], public_ip)
        if new_cand != inner["candidate"]:
            inner["candidate"] = new_cand
            modified = True

    if modified:
        data["peer_msg"]["msg"] = json.dumps(inner)
        return json.dumps(data)

    return raw


async def proxy_connection(browser_ws, kit_host: str, kit_port: int, public_ip: str):
    """Proxy a single WebSocket connection between browser and Kit."""
    # Forward the full path + query string from the browser's connection.
    # The websockets library stores the request target differently across versions:
    # - Legacy: browser_ws.path contains full path including query string
    # - New: browser_ws.request.path may strip query string
    # We try multiple attributes to get the full URI.
    full_path = None
    if hasattr(browser_ws, 'path'):
        # Legacy websockets API: .path includes query string
        full_path = browser_ws.path
    if not full_path and hasattr(browser_ws, 'request'):
        req = browser_ws.request
        full_path = getattr(req, 'path', None)
        # Some versions split path and query
        if full_path and '?' not in full_path:
            qs = getattr(req, 'query_string', None) or getattr(req, 'query', None)
            if qs:
                full_path = f"{full_path}?{qs}"
    if not full_path:
        full_path = "/sign_in"

    kit_uri = f"ws://{kit_host}:{kit_port}{full_path}"
    log.info("New connection: %s → %s", browser_ws.remote_address, kit_uri)

    try:
        async with websockets.connect(
            kit_uri,
            close_timeout=5,
            max_size=2**20,
        ) as kit_ws:
            async def browser_to_kit():
                """Forward browser messages to Kit (no rewriting needed)."""
                try:
                    async for msg in browser_ws:
                        log.info("Browser → Kit: %s", msg[:200] if isinstance(msg, str) else f"<binary {len(msg)} bytes>")
                        await kit_ws.send(msg)
                except websockets.ConnectionClosed as e:
                    log.info("Browser WS closed: code=%s reason=%s", e.code, e.reason)
                except Exception as e:
                    log.error("browser_to_kit error: %s", e)

            async def kit_to_browser():
                """Forward Kit messages to browser, rewriting SDP/ICE."""
                msg_count = 0
                try:
                    async for msg in kit_ws:
                        msg_count += 1
                        rewritten = rewrite_message(msg, public_ip)
                        log.info("Kit → Browser [msg %d, len=%d]", msg_count, len(rewritten))
                        await browser_ws.send(rewritten)
                except websockets.ConnectionClosed as e:
                    log.info("Kit WS closed after %d msgs: code=%s reason=%s", msg_count, e.code, e.reason)
                except Exception as e:
                    log.error("kit_to_browser error after %d msgs: %s", msg_count, e)

            # Run both directions concurrently
            done, pending = await asyncio.wait(
                [
                    asyncio.create_task(browser_to_kit()),
                    asyncio.create_task(kit_to_browser()),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()

    except (ConnectionRefusedError, OSError) as e:
        log.error("Cannot connect to Kit signaling at %s: %s", kit_uri, e)
    except Exception as e:
        log.error("Proxy error: %s", e)

    log.info("Connection closed: %s", browser_ws.remote_address)


async def main(args):
    log.info(
        "Starting signaling proxy: listen=%s:%d → kit=%s:%d, publicIp=%s",
        args.listen_host, args.listen_port,
        args.kit_host, args.kit_port,
        args.public_ip,
    )

    async def handler(ws):
        await proxy_connection(ws, args.kit_host, args.kit_port, args.public_ip)

    stop = asyncio.get_event_loop().create_future()

    def shutdown():
        if not stop.done():
            stop.set_result(True)

    for sig in (signal.SIGTERM, signal.SIGINT):
        asyncio.get_event_loop().add_signal_handler(sig, shutdown)

    async with ws_serve(
        handler,
        args.listen_host,
        args.listen_port,
        max_size=2**20,
        process_request=None,
    ):
        log.info("Proxy ready on %s:%d", args.listen_host, args.listen_port)
        await stop

    log.info("Proxy shut down")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebRTC signaling proxy with SDP rewriting")
    parser.add_argument("--public-ip", required=True, help="Public IP to inject into SDP")
    parser.add_argument("--listen-host", default="0.0.0.0", help="Host to listen on")
    parser.add_argument("--listen-port", type=int, default=49200, help="Port to listen on")
    parser.add_argument("--kit-host", default="127.0.0.1", help="Kit signaling host")
    parser.add_argument("--kit-port", type=int, default=49100, help="Kit signaling port")
    args = parser.parse_args()
    asyncio.run(main(args))
