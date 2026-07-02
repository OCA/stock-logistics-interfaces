#!/usr/bin/python3
import argparse
import asyncio
import logging
import os
import random
import ssl
import time
from dataclasses import dataclass, replace
from enum import IntEnum

import aiohttp

_logger = logging.getLogger(__name__)


class LoopBound:
    """Mixin: a one-time bindable event loop, held in ``self._loop``."""

    _loop = None

    def bind_loop(self, loop, silent=False):
        """Bind the event loop and return it.

        Idempotent if ``loop`` is the one already bound. Raises if a *different*
        loop is already bound, unless ``silent`` is set (then it is ignored).
        """
        if self._loop is loop:
            return self._loop
        if self._loop is not None:
            if silent:
                return self._loop
            raise RuntimeError("event loop already bound")
        self._loop = loop
        return self._loop


class KardexCode(IntEnum):
    """JMIF message ``code`` (field 1 of the pipe protocol)."""

    PICK = 1
    PUT = 2
    INVENTORY = 5
    PING = 61


@dataclass
class KardexShuttleAddr:
    """A JMIF address: ``<shuttle>-<gate>`` (e.g. ``SH8-1``)."""

    shuttle: str
    gate: int = 1

    @classmethod
    def parse(cls, text):
        # rpartition: the shuttle name itself may contain a '-'
        shuttle, sep, gate = text.rpartition("-")
        return cls(shuttle, int(gate))

    def __str__(self):
        return f"{self.shuttle}-{self.gate}"


@dataclass
class KardexMsg:
    """A JMIF pipe-protocol message; parses from and dumps to the wire format.

    Wire layout (12 ``|``-separated fields, trailing ``|`` then CRLF)::

        code|hostId|addr|carrier|carrierNext|x|y|boxType|Q|order|part|desc|
    """

    # NOTE: field meanings below are partly inferred from the Odoo integration,
    # not all confirmed against the Kardex JMIF spec (marked "?"). Verify before
    # relying on the uncertain ones.
    code: int  # message type: 1 pick / 2 put / 5 inv / 61 ping
    host_id: str  # request/correlation id, echoed back in the reply
    addr: KardexShuttleAddr  # target station + gate, e.g. SH8-1
    carrier: str = "0"  # ? tray/carrier number to move (0 = none/auto)
    carrier_next: str = "0"  # ? next tray to pre-stage for the following job
    x: str = ""  # ? column/bin X coordinate on the tray
    y: str = ""  # ? row/bin Y coordinate on the tray
    box_type: str = ""  # ? bin/box type on the tray
    quantity: str = ""  # ? quantity to pick/put (JMIF "Q")
    order: str = ""  # ? order / job reference
    part: str = ""  # ? article / part number
    desc: str = ""  # ? free-text description shown on operator display

    @classmethod
    def parse(cls, text):
        p = text.strip().split("|")
        return cls(
            code=int(p[0]),
            host_id=p[1],
            addr=KardexShuttleAddr.parse(p[2]),
            carrier=p[3],
            carrier_next=p[4],
            x=p[5],
            y=p[6],
            box_type=p[7],
            quantity=p[8],
            order=p[9],
            part=p[10],
            desc=p[11],
        )

    def dump(self):
        fields = [
            str(int(self.code)),
            self.host_id,
            str(self.addr),
            self.carrier,
            self.carrier_next,
            self.x,
            self.y,
            self.box_type,
            self.quantity,
            self.order,
            self.part,
            self.desc,
        ]
        return "|".join(fields) + "|\r\n"

    @classmethod
    def ping(cls, addr, host_id=None):
        """Factory for a keepalive ping message.

        ``addr`` may be a KardexShuttleAddr or a ``<shuttle>-<gate>`` string.
        """
        if not isinstance(addr, KardexShuttleAddr):
            addr = KardexShuttleAddr.parse(addr)
        if host_id is None:
            host_id = f"ping{int(time.time())}"
        return cls(code=KardexCode.PING, host_id=host_id, addr=addr)


class MessageStream:
    """An async-iterable stream of messages backed by a queue.

    Lets a callback-based protocol expose its messages to coroutines::

        async for msg in stream:
            ...
    """

    def __init__(self):
        self._queue = asyncio.Queue()

    def push(self, msg):
        self._queue.put_nowait(msg)

    def __aiter__(self):
        return self

    async def __anext__(self):
        return await self._queue.get()


class _InboundConnectionHandler(asyncio.Protocol):
    """One incoming (Odoo -> proxy) connection; pushes its message upstream."""

    def __init__(self, messages):
        _logger.info("Proxy: created")
        self.transport = None
        self.buffer = b""
        self._messages = messages

    def connection_made(self, transport):
        _logger.info("Proxy: incoming cnx made")
        self.transport = transport
        self.buffer = b""

    def data_received(self, data):
        self.buffer += data
        _logger.info("Proxy: received %s", data)
        if len(self.buffer) > 65535:
            # prevent buffer overflow
            self.transport.close()

    def eof_received(self):
        _logger.info("Proxy: received EOF")
        if self.buffer[-1] != b"\n":
            # bad format -> close
            self.transport.close()
        data = (
            self.buffer.replace(b"\r\n", b"\n")
            .replace(b"\n", b"\r\n")
            .decode("iso-8859-1", "replace")
        )
        self._messages.push(data)
        self.buffer = b""

    def connection_lost(self, exc):
        if exc:
            _logger.error("Proxy: incoming cnx lost: %s", exc)
        else:
            _logger.info("Proxy: incoming cnx closed")
        self.transport = None
        self.buffer = b""


class InboundServer(LoopBound):
    """Accepts raw-TCP connections from Odoo and exposes their messages.

    Usage::

        inbound = InboundServer(loop)
        await inbound.start(host, port)
        async for msg in inbound.messages:
            ...
    """

    def __init__(self, loop=None):
        self._messages = MessageStream()
        if loop is not None:
            self.bind_loop(loop)

    def messages(self):
        """Async-iterable of messages received from Odoo."""
        return self._messages

    def _handler(self):
        """Protocol factory: a fresh handler per incoming connection, all
        sharing this server's message stream."""
        return _InboundConnectionHandler(self._messages)

    async def start(self, host, port):
        await self._loop.create_server(self._handler, host=host, port=port)


class ReconnectingTCPClientProtocol(asyncio.Protocol, LoopBound):
    # source: https://stackoverflow.com/a/49452683/1504003
    max_delay = 3600
    initial_delay = 1.0
    factor = 2.7182818284590451
    jitter = 0.119626565582
    max_retries = None

    def __init__(self, *args, loop=None, **kwargs):
        # loop is optional; must be bound (bind_loop) before connect()
        if loop is not None:
            self.bind_loop(loop)
        self._args = args
        self._kwargs = kwargs
        self._retries = 0
        self._delay = self.initial_delay
        self._continue_trying = True
        self._call_handle = None
        self._connector = None

    def connection_lost(self, exc):
        if self._continue_trying:
            self.retry()

    def connection_failed(self, exc):
        if self._continue_trying:
            self.retry()

    def retry(self):
        if not self._continue_trying:
            return

        self._retries += 1
        if self.max_retries is not None and (self._retries > self.max_retries):
            self.stop_trying()
            return

        self._delay = min(self._delay * self.factor, self.max_delay)
        if self.jitter:
            self._delay = random.normalvariate(self._delay, self._delay * self.jitter)
        _logger.info("%s: will retry connection after %ss", self, self._delay)
        self._call_handle = self._loop.call_later(self._delay, self.connect)

    def connect(self):
        if self._connector is None:
            self._connector = self._loop.create_task(self._connect())

    async def _connect(self):
        try:
            await self._loop.create_connection(
                lambda: self, *self._args, **self._kwargs
            )
        except Exception as exc:
            self._loop.call_soon(self.connection_failed, exc)
        else:
            self._delay = self.initial_delay
            self._retries = 0
        finally:
            self._connector = None

    def stop_trying(self):
        if self._call_handle:
            self._call_handle.cancel()
            self._call_handle = None
        self._continue_trying = False
        if self._connector is not None:
            self._connector.cancel()
            self._connector = None


class _KardexConnection(ReconnectingTCPClientProtocol):
    """Low-level: the raw (auto-reconnecting) socket to the Kardex server.

    Handles framing and request/reply correlation; knows nothing about pings
    or Odoo. Replies with a matching pending request wake that request; all
    other messages are pushed to ``unsolicited_events``.
    """

    max_delay = 15
    initial_delay = 0.5
    factor = 1.7182818284590451
    jitter = 0.119626565582
    # if we set a number of retries, after N failed
    # retries, it will stop the event loop and exit
    max_retries = None

    def __init__(self, host, port, loop=None, ssl_context=None):
        super().__init__(loop=loop, host=host, port=port, ssl=ssl_context)
        _logger.info("started kardex connection")
        self.transport = None
        self.buffer = b""
        # in-flight requests awaiting a reply, keyed by the message key
        # (field 2 of the pipe protocol)
        self._pending = {}
        # spontaneous, Kardex-initiated messages (not a reply to a request)
        self.unsolicited_events = MessageStream()

    def ensure_connected(self):
        # connect only if not connected AND no (re)connect already in flight,
        # otherwise we race ReconnectingTCPClientProtocol's scheduled retry
        if (
            self.transport is None
            and self._connector is None
            and self._call_handle is None
        ):
            self.connect()

    def reset(self):
        # force-close a (possibly half-open) link; connection_lost then
        # schedules a reconnect via the base class
        if self.transport is not None:
            self.transport.close()

    def connection_made(self, transport):
        self.transport = transport
        _logger.info("connected to kardex server %r", transport)

    @staticmethod
    def _key(message):
        # correlation key = field 2 of the pipe protocol (e.g. ``L798279``,
        # ``ping1782129339``), echoed by Kardex in the matching reply
        parts = message.split("|")
        return parts[1] if len(parts) > 1 else ""

    async def request(self, message, timeout=None):
        """Send a message and await its correlated reply.

        ``message`` may be a KardexMsg or an already-serialized wire string.
        Registers a future under the message key so ``data_received`` can wake
        us with the matching reply. Raises ``ConnectionError`` if the link is
        down, ``asyncio.TimeoutError`` if no reply arrives within ``timeout``.
        """
        if isinstance(message, KardexMsg):
            message = message.dump()
        key = self._key(message)
        fut = self._loop.create_future()
        self._pending[key] = fut
        try:
            await self._send(message)
            return await asyncio.wait_for(fut, timeout)
        finally:
            self._pending.pop(key, None)

    async def _send(self, message):
        _logger.info("SEND %r", message)
        if self.transport is None:
            # connection is down (reconnect in progress): don't write into a
            # missing transport, let the caller handle the failure
            raise ConnectionError("no transport, kardex connection is down")
        self.transport.write(message.encode("iso-8859-1"))

    def data_received(self, data):
        data = data.replace(b"\0", b"")
        _logger.info("RECV %s", data)
        self.buffer += data
        # a single data_received may carry several framed messages (or none)
        while b"\r\n" in self.buffer:
            msg, sep, self.buffer = self.buffer.partition(b"\r\n")
            self._dispatch(msg.decode("iso-8859-1", "replace").strip())

    def _dispatch(self, msg):
        fut = self._pending.get(self._key(msg))
        if fut is not None and not fut.done():
            # correlated reply -> wake the coroutine that sent the request
            fut.set_result(msg)
        else:
            # unsolicited Kardex-initiated message -> expose to the consumer
            _logger.info("unsolicited from kardex: %s", msg)
            self.unsolicited_events.push(msg)

    def connection_lost(self, exc):
        _logger.error("Kardex client: connection lost: %s", exc)
        # fail every in-flight request so their awaiters (commands, ping) wake
        # up with an error instead of hanging until their own timeout
        self._fail_pending(exc or ConnectionError("connection lost"))
        return super().connection_lost(exc)

    def _fail_pending(self, exc):
        if not isinstance(exc, BaseException):
            exc = ConnectionError(str(exc))
        for fut in self._pending.values():
            if not fut.done():
                fut.set_exception(exc)
        self._pending.clear()

    def connection_failed(self, exc):
        _logger.error("Kardex client: failed to open connection: %s", exc)
        return super().connection_failed(exc)

    def stop_trying(self):
        res = super().stop_trying()
        self._loop.stop()
        return res


@dataclass
class KardexClientOptions:
    """Tuning for KardexClient's ping/keepalive watchdog (seconds)."""

    initial_keepalive_delay: int = 20
    ping_interval: int = 50
    ping_max_failures: int = 2
    ping_timeout: int = 10
    command_timeout: int = 30


class KardexClient(LoopBound):
    """High-level: keeps the Kardex link healthy and exposes a clean API.

    Owns the low-level connection plus the ping/keepalive policy::

        client = KardexClient(host=..., port=...)
        loop.create_task(client.keepalive())
        answer = await client.request(command)
        async for event in client.kardex_events():
            ...
    """

    def __init__(self, host, port, options=None, loop=None, ssl_context=None):
        # copy so later mutation of the caller's instance has no side effect
        self._options = replace(options) if options else KardexClientOptions()
        self._missed_pings = 0
        # per-shuttle liveness; None until keepalive() starts (only one allowed)
        self._alive = None
        self._connection = _KardexConnection(host, port, ssl_context=ssl_context)
        if loop is not None:
            self.bind_loop(loop)

    def bind_loop(self, loop, silent=False):
        res = super().bind_loop(loop, silent)
        # keep the underlying connection on the same loop
        self._connection.bind_loop(loop, silent=True)
        return res

    @property
    def command_timeout(self):
        return self._options.command_timeout

    def kardex_events(self):
        """Async-iterable of spontaneous, Kardex-initiated messages."""
        return self._connection.unsolicited_events

    async def request(self, message, timeout=None):
        return await self._connection.request(message, timeout)

    def _check_link(self, max_failures):
        """One health check of the shared link: reconnect if all shuttles down."""
        # ensure_connected() self-heals if the link is down with no
        # reconnect pending (guarded, so no race with the scheduled retry)
        self._connection.ensure_connected()
        if any(self._alive.values()):
            self._missed_pings = 0
            return
        # every shuttle is down -> the shared link itself is broken
        self._missed_pings += 1
        _logger.warning(
            "Kardex client: %d/%d consecutive link failure(s)",
            self._missed_pings,
            max_failures,
        )
        if self._missed_pings >= max_failures:
            _logger.error(
                "Kardex client: connection considered broken, forcing reconnect"
            )
            self._missed_pings = 0
            self._connection.reset()

    async def keepalive(self, shuttles):
        """Monitor the shared link's health from per-shuttle ping loops.

        Spawns one ``_keepalive`` per shuttle (each tracks its own shuttle's
        liveness in ``self._alive``) and watches the aggregate: the shared
        socket is alive as long as *any* shuttle answers, so a single down
        shuttle does not trigger a reconnect. Only when *every* shuttle is down
        is the link deemed broken and reconnected.

        We cannot rely on the OS to detect a dead/half-open connection: on a
        silently dropped peer the kernel only reports the loss after
        ``tcp_retries2`` retransmissions (~15 min by default), hence this
        application-level probe.
        """
        if self._alive is not None:
            raise RuntimeError("keepalive already running for this KardexClient")
        opts = self._options
        self._alive = {s: True for s in shuttles}
        self._connection.ensure_connected()
        if not shuttles:
            # nothing to monitor: connection still opens (and the base class
            # reconnects on OS-detected loss), but the half-open watchdog is off
            _logger.warning(
                "Kardex client: no shuttles to monitor; keepalive watchdog disabled"
            )
            return
        for shuttle in shuttles:
            self._loop.create_task(self._keepalive(shuttle))
        await asyncio.sleep(opts.initial_keepalive_delay)
        while True:
            self._check_link(opts.ping_max_failures)
            await asyncio.sleep(opts.ping_interval)

    async def _keepalive(self, shuttle):
        """Ping one shuttle forever, tracking its up/down state in self._alive."""
        await asyncio.sleep(self._options.initial_keepalive_delay)
        while True:
            ok = await self.ping(shuttle)
            if ok != self._alive.get(shuttle, True):
                state = "up" if ok else "down"
                _logger.warning("Kardex: shuttle %s is now %s", shuttle, state)
            self._alive[shuttle] = ok
            await asyncio.sleep(self._options.ping_interval)

    async def ping(self, addr):
        """Send one keepalive ping. Return True if Kardex answered in time.

        ``addr`` may be a KardexShuttleAddr or a ``<shuttle>-<gate>`` string.
        """
        try:
            await self._connection.request(
                KardexMsg.ping(addr), self._options.ping_timeout
            )
            _logger.info("ping ok")
            return True
        except (asyncio.TimeoutError, ConnectionError) as exc:
            _logger.warning("Kardex client: ping failed: %s", exc)
            return False


class OdooNotifier:
    """Posts Kardex outcomes back to Odoo's ``/vertical-lift`` endpoint.

    Reuses a single ``aiohttp.ClientSession`` so its connection pool keeps
    HTTP keep-alive connections to Odoo alive across calls.
    """

    def __init__(self, odoo_url, secret):
        self._url = odoo_url + "/vertical-lift"
        self._secret = secret
        self._session = None

    def _get_session(self):
        # created lazily: needs a running event loop
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self):
        if self._session is not None:
            await self._session.close()

    async def _post(self, params):
        # Retry once on a connection error: after an Odoo restart the pooled
        # keep-alive connection is stale and the first POST fails; dropping the
        # session forces a fresh connection on the retry.
        for attempt in (1, 2):
            try:
                async with self._get_session().post(self._url, data=params) as resp:
                    resp_text = await resp.text()
                    _logger.info("Reponse from Odoo: %s %s", resp.status, resp_text)
                    return
            except aiohttp.ClientConnectionError as exc:
                # stale connection -> drop session, retry once with a fresh one
                _logger.warning("Odoo post failed (attempt %d): %s", attempt, exc)
                await self.close()
                self._session = None
            except Exception as exc:
                # never let a post failure crash the caller's task/loop
                _logger.error("Odoo post error: %s", exc)
                return
        _logger.error("Giving up posting to Odoo after retry: %r", params)

    async def send_answer(self, answer):
        await self._post({"answer": answer, "secret": self._secret})

    async def send_error(self, command, error):
        # Report a failure to Odoo so it lands in the ERROR column of the
        # matching ``vertical.lift.command`` (matched by the command's key).
        # NOTE: requires the ``/vertical-lift`` controller to accept the
        # ``command`` + ``error`` parameters (deferred change, separate PR);
        # no-op with the current controller.
        _logger.info("Reporting error to Odoo for %r: %s", command, error)
        await self._post({"command": command, "error": error, "secret": self._secret})


class KardexProxy(LoopBound):
    """Mediator/Orchestrator wiring the Odoo and Kardex sides together.

    The collaborators are injected; the mediator only coordinates them::

        proxy = KardexProxy(kardex_client, odoo_notifier, shuttles=["SH8-1"])
        proxy.run(host, port)
    """

    def __init__(
        self, kardex_client, odoo_notifier, shuttles=None, odoo_inbound=None, loop=None
    ):
        self.kardex_client = kardex_client
        self.odoo_notifier = odoo_notifier
        # InboundServer needs no config, so default it here
        self.odoo_inbound = odoo_inbound or InboundServer()
        # shuttles to keepalive-monitor; accept KardexShuttleAddr or "SH-gate"
        # (may be empty -> nothing is monitored, see KardexClient.keepalive)
        self.shuttles = [
            s if isinstance(s, KardexShuttleAddr) else KardexShuttleAddr.parse(s)
            for s in (shuttles or [])
        ]
        if loop is not None:
            self.bind_loop(loop)

    def bind_loop(self, loop, silent=False):
        res = super().bind_loop(loop, silent)
        # align the collaborators onto the same loop; tolerate ones already
        # bound (to this same loop or intentionally elsewhere)
        self._bind_all(self._loop, silent=True)
        return res

    def _bind_all(self, loop, silent=False):
        self.odoo_inbound.bind_loop(loop, silent=silent)
        self.kardex_client.bind_loop(loop, silent=silent)

    def _get_loop(self):
        # adopt the default loop only if none was injected
        if self._loop is None:
            self.bind_loop(asyncio.get_event_loop())
        return self._loop

    def run(self, host, port):
        loop = self._get_loop()
        loop.run_until_complete(self.odoo_inbound.start(host, port))
        # keepalive owns the initial connect (and all reconnects)
        loop.create_task(self.kardex_client.keepalive(self.shuttles))
        loop.create_task(self._pump_commands())
        loop.create_task(self._forward_notifications())
        loop.run_forever()

    async def _pump_commands(self):
        # each command runs concurrently; replies are de-multiplexed by key
        async for message in self.odoo_inbound.messages():
            self._loop.create_task(self._handle_command(message))

    async def _handle_command(self, message):
        """Forward one Odoo command to Kardex and report the outcome back."""
        try:
            answer = await self.kardex_client.request(
                message, self.kardex_client.command_timeout
            )
        except (asyncio.TimeoutError, ConnectionError) as exc:
            _logger.error("Kardex client: command %r failed: %s", message, exc)
            # no reply (timeout/drop): surface it in the command ERROR column
            await self.odoo_notifier.send_error(message, f"Kardex error: {exc}")
            return
        await self.odoo_notifier.send_answer(answer)

    async def _forward_notifications(self):
        async for message in self.kardex_client.kardex_events():
            await self.odoo_notifier.send_answer(message)


def make_parser():
    listen_address = os.environ.get("INTERFACE", "0.0.0.0")
    listen_port = int(os.environ.get("PORT", "7654"))
    secret = os.environ.get("ODOO_CALLBACK_SECRET", "")
    odoo_url = os.environ.get("ODOO_URL", "http://localhost:8069")
    odoo_db = os.environ.get("ODOO_DB", "odoodb")
    kardex_host = os.environ.get("KARDEX_HOST", "kardex")
    kardex_port = int(os.environ.get("KARDEX_PORT", "9600"))
    kardex_use_tls = (
        False
        if os.environ.get("KARDEX_TLS", "") in ("", "0", "false", "False", "FALSE")
        else True
    )
    debug = (
        True if os.environ.get("DEBUG", "") in ("1", "true", "True", "TRUE") else False
    )
    # ping watchdog tuning
    # KARDEX_PING_INTERVAL: seconds between two pings to the kardex server
    ping_interval = int(os.environ.get("KARDEX_PING_INTERVAL", "50"))
    # KARDEX_PING_MAX_FAILURES: number of consecutive unanswered pings after
    # which the connection is considered broken and a reconnect is forced
    ping_max_failures = int(os.environ.get("KARDEX_PING_MAX_FAILURES", "2"))
    # KARDEX_PING_INITIAL_DELAY: seconds to wait before the first ping
    ping_initial_delay = int(os.environ.get("KARDEX_PING_INITIAL_DELAY", "20"))
    # KARDEX_PING_TIMEOUT: seconds to wait for a ping reply before it counts
    # as a failure
    ping_timeout = int(os.environ.get("KARDEX_PING_TIMEOUT", "10"))
    # KARDEX_COMMAND_TIMEOUT: seconds to wait for a command reply before
    # reporting it as errored to Odoo
    command_timeout = int(os.environ.get("KARDEX_COMMAND_TIMEOUT", "30"))
    # KARDEX_SHUTTLES: comma-separated shuttle addresses to keepalive-monitor
    shuttles = os.environ.get("KARDEX_SHUTTLES", "SH8-1")
    parser = argparse.ArgumentParser()
    arguments = [
        ("--host", listen_address, str),
        ("--port", listen_port, int),
        ("--odoo-url", odoo_url, str),
        ("--odoo-db", odoo_db, str),
        ("--secret", secret, str),
        ("--kardex-host", kardex_host, str),
        ("--kardex-port", kardex_port, str),
        ("--kardex-use-tls", kardex_use_tls, bool),
        ("--kardex-ping-interval", ping_interval, int),
        ("--kardex-ping-max-failures", ping_max_failures, int),
        ("--kardex-ping-initial-delay", ping_initial_delay, int),
        ("--kardex-ping-timeout", ping_timeout, int),
        ("--kardex-command-timeout", command_timeout, int),
        ("--kardex-shuttles", shuttles, str),
        ("--debug", debug, bool),
    ]
    for name, default, type_ in arguments:
        parser.add_argument(name, default=default, action="store", type=type_)
    return parser


def main(args=None, ssl_context=None):
    # backward-compatible: callable as main() (parses env/CLI itself) or as the
    # old main(args, ssl_context=None) with pre-parsed args / an injected context
    if args is None:
        args = make_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    kardex_options = KardexClientOptions(
        initial_keepalive_delay=args.kardex_ping_initial_delay,
        ping_interval=args.kardex_ping_interval,
        ping_max_failures=args.kardex_ping_max_failures,
        ping_timeout=args.kardex_ping_timeout,
        command_timeout=args.kardex_command_timeout,
    )
    if not args.kardex_use_tls:
        _logger.info("TLS disabled")
        ssl_context = None
    elif not ssl_context:
        ssl_context = ssl.create_default_context()

    loop = asyncio.get_event_loop()
    loop.set_debug(args.debug)

    # composition root: build the collaborators and wire them together
    shuttles = [s.strip() for s in args.kardex_shuttles.split(",") if s.strip()]
    proxy = KardexProxy(
        KardexClient(
            args.kardex_host,
            args.kardex_port,
            options=kardex_options,
            ssl_context=ssl_context,
        ),
        OdooNotifier(args.odoo_url, args.secret),
        shuttles=shuttles,
        loop=loop,
    )
    proxy.run(args.host, args.port)


if __name__ == "__main__":
    main()
