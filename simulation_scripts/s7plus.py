"""
=============================================================================
NATIVE S7COMMPLUS PLC SERVER - GENUINE WIRESHARK-RECOGNIZED PROTOCOL
=============================================================================
Implements actual S7commPlus wire format as documented in Wireshark's
s7comm_plus dissector (reverse-engineered from Siemens TIA Portal traffic):

PROTOCOL STRUCTURE (on-the-wire):
  TPKT Header (4 bytes):
    [0x03][0x00][length_hi][length_lo]  # ISO 8073 transport layer
  
  COTP Header (3 bytes):
    [0x02][0xF0][0x01]  # Data PDU, parameter code, length
  
  S7COMMPLUS Message:
    [0x72]                    # Protocol ID (magic byte)
    [version]                 # 0x01 (currently)
    [reserved]                # 0x00
    [sequence_hi][sequence_lo] # Message sequence number
    [opcode]                  # Request(0x22), Response(0x32), Notification(0x62)
    [reserved2]               # 0x00
    [function_hi][function_lo] # Function code (e.g., 0x0001=CreateObject)
    [length_hi][length_lo]    # Data length (payload)
    [function_id]             # Function instance ID
    [data_len_hi][data_lo]    # Payload length
    ... [payload] ...
    [0x72]                    # Protocol ID (trailer - frame validation)
    [function_status]         # 0x00 = OK, non-zero = error

Addressing Model:
  - Real S7+ uses compiled symbol IDs from TIA's symbol table
  - We generate stable CRC32(symbol_name) IDs for determinism
  - Requests reference symbols by ID, not DB+offset like classic S7comm
  - Responses carry symbol values
  
Subscription Model:
  - Unlike classic S7comm polling, S7+ uses async notifications
  - HMI subscribes to variable changes via CreateSubscription
  - PLC sends unsolicited Notification messages when values change
  - This is the real TIA Portal / modern Siemens architecture
=============================================================================
"""

import asyncio
import random
import struct
import threading
import time
import logging
import numpy as np
import zlib

# ============================================================================
# LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("S7PLUS")

# ============================================================================
# CONFIGURATION
# ============================================================================
SERVER_IP = "127.0.0.1"
SERVER_PORT = 102
DB_SIZE = 256

S7PLUS_PROTOCOL_ID = 0x72
S7PLUS_VERSION = 0x01  # kept for simulator compatibility; wire PDU type is used below

# S7+ Opcodes
OPCODE_REQUEST = 0x31
OPCODE_RESPONSE = 0x32
OPCODE_NOTIFICATION = 0x33

# S7+ Functions
FUNC_CREATE_OBJECT = 0x04CA
FUNC_DELETE_OBJECT = 0x04D4
FUNC_CREATE_SUBSCRIPTION = 0x0064
FUNC_DELETE_SUBSCRIPTION = 0x0065
FUNC_MODIFY_SUBSCRIPTION = 0x0066
FUNC_TRIGGER_VARIABLE_UPDATE = 0x0012

# Physics Configuration
PHYSICS_UPDATE_INTERVAL = 0.05
SINE_FREQUENCY = 0.1
ANOMALY_INJECTION_RATE = 0.001

PRESSURE_BASE = 0.53
TEMP_BASE = 23.5
SPEED_BASE = 1.35
RPM_BASE = 1450
FLOW_RATE_BASE = 15.5

running = True


# ============================================================================
# SYMBOL TABLE & CRC32 ADDRESSING
# ============================================================================
class SymbolTable:
    """Map variable names to stable CRC32-based symbol IDs."""
    
    SYMBOLS = {
        "Pression": ("pression", ">f"),
        "TempFour": ("temp_four", ">f"),
        "VitesseConv": ("vitesse_conv", ">f"),
        "MotorRPM": ("rpm", ">h"),
        "FlowRate": ("flow_rate", ">f"),
        "SystemUptime": ("uptime_seconds", ">i"),
        "CycleCount": ("cycle_count", ">i"),
        "MachineState": ("machine_state", ">B"),
        "Conveyor": ("conveyor", ">B"),
        "PaintValve": ("paint_valve", ">B"),
        "EjectValve": ("eject_valve", ">B"),
    }
    
    @staticmethod
    def symbol_to_id(symbol_name: str) -> int:
        return zlib.crc32(symbol_name.encode()) & 0xFFFFFFFF
    
    @staticmethod
    def get_all_ids() -> dict:
        return {name: SymbolTable.symbol_to_id(name) for name in SymbolTable.SYMBOLS.keys()}


# ============================================================================
# PROCESS SIMULATOR
# ============================================================================
class ProcessSimulator:
    """Realistic industrial process simulator."""

    def __init__(self):
        self.time_accumulator = 0.0
        self.anomaly_active = False
        self.anomaly_start = 0.0
        self.anomaly_duration = 0.0

        self.pression = PRESSURE_BASE
        self.temp_four = TEMP_BASE
        self.vitesse_conv = SPEED_BASE
        self.rpm = 0.0
        self.flow_rate = FLOW_RATE_BASE
        self.machine_state = 0
        self.cycle_count = 0
        self.uptime_seconds = 0.0
        self.conveyor = False
        self.paint_valve = False
        self.eject_valve = False

    def update(self, dt: float, machine_state: int):
        self.time_accumulator += dt
        self.uptime_seconds += dt
        self.machine_state = machine_state

        if random.random() < ANOMALY_INJECTION_RATE:
            self._trigger_anomaly()

        if self.anomaly_active:
            if (time.time() - self.anomaly_start) > self.anomaly_duration:
                self.anomaly_active = False
            else:
                self._apply_anomaly()

        if machine_state == 0:      # Idle
            base_p, base_t, base_v, base_rpm = PRESSURE_BASE * 0.8, TEMP_BASE, 0.0, 0.0
        elif machine_state == 1:    # Running
            base_p, base_t, base_v, base_rpm = PRESSURE_BASE, TEMP_BASE, SPEED_BASE, RPM_BASE
        elif machine_state == 2:    # Painting
            base_p, base_t, base_v, base_rpm = 4.40, 25.6, 0.0, 0.0
        elif machine_state == 3:    # QC
            base_p, base_t, base_v, base_rpm = PRESSURE_BASE * 0.9, TEMP_BASE, 0.0, 0.0
        elif machine_state == 4:    # Eject
            base_p, base_t, base_v, base_rpm = 1.18, 24.1, 0.0, 500.0
        else:
            base_p, base_t, base_v, base_rpm = PRESSURE_BASE, TEMP_BASE, SPEED_BASE, RPM_BASE

        sine = 0.05 * np.sin(2 * np.pi * SINE_FREQUENCY * self.time_accumulator)

        self.pression = max(0.0, base_p + sine + random.gauss(0, 0.05))
        self.temp_four = max(0.0, base_t + sine * 2 + random.gauss(0, 0.3))
        self.vitesse_conv = max(0.0, base_v + random.gauss(0, 0.03))
        self.rpm = max(0.0, base_rpm + random.gauss(0, 15))
        self.flow_rate = max(0.0, FLOW_RATE_BASE + sine + random.gauss(0, 0.4))

    def _trigger_anomaly(self):
        anomaly_type = random.choice(["pressure_spike", "temp_drop", "speed_loss"])
        self.anomaly_active = True
        self.anomaly_start = time.time()
        self.anomaly_duration = random.uniform(2.0, 8.0)
        logger.warning(f"[ANOMALY] {anomaly_type} ({self.anomaly_duration:.1f}s)")

    def _apply_anomaly(self):
        elapsed = time.time() - self.anomaly_start
        progress = elapsed / self.anomaly_duration
        if progress < 0.3:
            effect = progress / 0.3
        elif progress < 0.7:
            effect = 1.0
        else:
            effect = (1.0 - progress) / 0.3
        self.pression += random.gauss(0.4, 0.2) * effect
        self.temp_four += random.gauss(2.5, 1.0) * effect
        self.rpm += random.gauss(150, 40) * effect


# ============================================================================
# S7COMMPLUS PACKET BUILDER/PARSER
# ============================================================================
def build_s7plus_message(opcode: int, function: int, sequence: int,
                         function_id: int, payload: bytes = b"") -> bytes:
    """
    Build an S7comm-Plus PDU in the layout expected by the Wireshark
    S7COMM-PLUS dissector.

    Wire layout:
        TPKT (4) + COTP (3)
        S7+ header:
            Protocol ID  1 byte = 0x72
            PDU type     1 byte = 0x02 (Data)
            Data length  2 bytes, big endian
        Data:
            Opcode       1 byte
            Reserved     2 bytes
            Function     2 bytes
            Reserved     2 bytes
            Sequence     2 bytes
            Unknown      1 byte
            Session ID   4 bytes
            Application payload
        S7+ trailer:
            Protocol ID  1 byte = 0x72
            PDU type     1 byte
            Data length  2 bytes

    The first four S7+ bytes and the data header follow the field layout
    used by the Wireshark S7COMM-PLUS dissector.
    """
    # The Wireshark dissector defines:
    #   0x72 = S7COMM-PLUS protocol identifier
    #   0x02 = Data PDU
    pdu_type = 0x02

    # Keep a stable session for this simulated PLC/HMI conversation.
    # A real S7comm-Plus session negotiates this value during connection setup.
    session_id = 0x00000001

    # S7comm-Plus data header fields exposed by Wireshark:
    # opcode(1) + reserved1(2) + function(2) + reserved2(2)
    # + sequence(2) + unknown1(1) + session_id(4) = 14 bytes.
    data_header = struct.pack(
        ">BHHHHBI",
        opcode,
        0x0000,          # reserved1
        function,
        0x0000,          # reserved2
        sequence & 0xFFFF,
        0x00,            # unknown1 / flags
        session_id
    )

    # Keep the application's function_id in the payload so the existing
    # simulator/HMI logic can continue to correlate operations.
    application_payload = bytes([function_id & 0xFF]) + payload

    data_block = data_header + application_payload
    data_length = len(data_block)

    # S7comm-Plus header + data + trailer.
    s7_header = struct.pack(
        ">BBH",
        S7PLUS_PROTOCOL_ID,   # 0x72
        pdu_type,             # 0x02 = Data
        data_length
    )

    s7_trailer = struct.pack(
        ">BBH",
        S7PLUS_PROTOCOL_ID,   # 0x72
        pdu_type,             # 0x02
        data_length
    )

    s7_message = s7_header + data_block + s7_trailer

    # TPKT length includes TPKT + COTP + S7+ message.
    tpkt_length = 4 + 3 + len(s7_message)

    tpkt_header = struct.pack(
        ">BBH",
        0x03,                 # TPKT version
        0x00,
        tpkt_length
    )

    # COTP Data TPDU:
    # 02 = header length indicator
    # F0 = DT TPDU
    # 80 = EOT
    cotp_header = b"\x02\xF0\x80"

    return tpkt_header + cotp_header + s7_message


def parse_s7plus_message(raw_bytes: bytes) -> tuple:
    """Parse the same S7comm-Plus layout generated above."""
    if len(raw_bytes) < 4 + 3 + 4 + 14 + 4:
        return None

    # Skip TPKT + COTP.
    msg = raw_bytes[7:]

    # S7+ header: protocol ID, PDU type, data length.
    protocol_id, pdu_type, data_length = struct.unpack(">BBH", msg[:4])

    if protocol_id != S7PLUS_PROTOCOL_ID:
        return None

    if pdu_type not in (0x01, 0x02, 0x03, 0xFF):
        return None

    if len(msg) < 4 + data_length + 4:
        return None

    data = msg[4:4 + data_length]

    # Keep-alive is a special PDU and does not contain the normal data header.
    if pdu_type == 0xFF:
        return (None, None, 0, 0, b"")

    if len(data) < 14:
        return None

    opcode = data[0]
    function = struct.unpack(">H", data[3:5])[0]
    sequence = struct.unpack(">H", data[7:9])[0]

    # Our application payload begins after the 14-byte S7+ data header.
    application = data[14:]

    if not application:
        function_id = 0
        payload = b""
    else:
        function_id = application[0]
        payload = application[1:]

    # Validate the S7+ trailer.
    trailer = msg[4 + data_length:4 + data_length + 4]
    trailer_protocol, trailer_pdu_type, trailer_length = struct.unpack(
        ">BBH", trailer
    )

    if (
        trailer_protocol != S7PLUS_PROTOCOL_ID
        or trailer_pdu_type != pdu_type
        or trailer_length != data_length
    ):
        logger.warning(
            "[PARSE] Invalid S7comm-Plus trailer: "
            f"{trailer.hex(' ')}"
        )
        return None

    return opcode, function, sequence, function_id, payload


# ============================================================================
# PLC SERVER ENGINE
# ============================================================================
class S7PlusPLC:
    def __init__(self):
        self.process_sim = ProcessSimulator()
        self.lock = threading.Lock()
        self.subscriptions = {}
        self.active_writers = {}
        self.next_subscription_id = 1
        self.next_sequence = 1

    def next_seq(self):
        self.next_sequence = (self.next_sequence + 1) & 0xFFFF
        return self.next_sequence


async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, plc: S7PlusPLC):
    peer = writer.get_extra_info("peername")
    logger.info(f"[SERVER] Client connected from {peer}")
    plc.active_writers[peer] = writer
    
    try:
        while running:
            try:
                tpkt = await reader.readexactly(4)
                length = struct.unpack(">H", tpkt[2:4])[0]
                rest = await reader.readexactly(length - 4)  # FIX: Subtract 4 TPKT header bytes
                full_msg = tpkt + rest
                
                result = parse_s7plus_message(full_msg)
                if not result:
                    continue
                
                opcode, function, sequence, function_id, payload = result
                
                if opcode == OPCODE_REQUEST:
                    if function == FUNC_CREATE_OBJECT:
                        resp = build_s7plus_message(OPCODE_RESPONSE, FUNC_CREATE_OBJECT, sequence, 0, b"\x00")
                        writer.write(resp)
                        await writer.drain()
                        logger.info("[SERVER] CreateObject ack")
                    
                    elif function == FUNC_CREATE_SUBSCRIPTION:
                        symbol_ids = []
                        for i in range(0, len(payload), 4):
                            sym_id = struct.unpack(">I", payload[i:i+4])[0]
                            symbol_ids.append(sym_id)
                        
                        sub_id = plc.next_subscription_id
                        plc.next_subscription_id += 1
                        plc.subscriptions[sub_id] = {"symbol_ids": symbol_ids, "peer": peer}
                        
                        resp = build_s7plus_message(OPCODE_RESPONSE, FUNC_CREATE_SUBSCRIPTION, sequence, 0, struct.pack(">H", sub_id))
                        writer.write(resp)
                        await writer.drain()
                        logger.info(f"[SERVER] Subscription {sub_id} created for {len(symbol_ids)} symbols")
                    
                    elif function == FUNC_DELETE_SUBSCRIPTION:
                        if len(payload) >= 2:
                            sub_id = struct.unpack(">H", payload[:2])[0]
                            plc.subscriptions.pop(sub_id, None)
                        resp = build_s7plus_message(OPCODE_RESPONSE, FUNC_DELETE_SUBSCRIPTION, sequence, 0, b"\x00")
                        writer.write(resp)
                        await writer.drain()
                        logger.info(f"[SERVER] Subscription deleted")
            
            except asyncio.IncompleteReadError:
                break
            except Exception as e:
                logger.error(f"[SERVER] Error handling client {peer}: {e}")
                break
    finally:
        plc.active_writers.pop(peer, None)
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass
        logger.info(f"[SERVER] Client disconnected: {peer}")


# ============================================================================
# PROCESS SIMULATION & NOTIFICATION DISPATCHER
# ============================================================================
def background_process_loop(plc: S7PlusPLC):
    logger.info("[PHYSICS] Background process simulation started (20 Hz)")
    
    # State sequence: (State ID, Name, Duration in Seconds)
    state_sequence = [
        (1, "RUNNING", 6.0),    # 6 seconds
        (2, "PAINTING", 15.0),  # 15 seconds
        (3, "QC", 8.0),         # 8 seconds
        (4, "EJECT", 10.0),     # 10 seconds
        (0, "IDLE", 4.0),       # 4 seconds
    ]
    
    current_idx = 0
    state_start_time = time.perf_counter()
    last_update = time.perf_counter()

    while running:
        try:
            now = time.perf_counter()
            dt = now - last_update
            last_update = now

            state_id, state_name, duration = state_sequence[current_idx]

            # Check if current state duration has elapsed
            if (now - state_start_time) >= duration:
                current_idx = (current_idx + 1) % len(state_sequence)
                state_start_time = now
                state_id, state_name, duration = state_sequence[current_idx]
                logger.info(f"[PROCESS] Transitioned to state: {state_name} ({duration}s)")

            # Update simulation physics
            plc.process_sim.update(dt, state_id)

            # Precise loop sleep for 20 Hz (0.05s interval)
            elapsed = time.perf_counter() - now
            sleep_time = max(0.0, PHYSICS_UPDATE_INTERVAL - elapsed)
            time.sleep(sleep_time)

        except Exception as e:
            logger.error(f"[PHYSICS] Error in process loop: {e}")
            time.sleep(0.1)


async def notification_sender(plc: S7PlusPLC):
    logger.info("[NOTIFY] Notification sender started")
    while running:
        await asyncio.sleep(1.0)  # Updated to 1Hz
        if plc.subscriptions and plc.active_writers:
            sim = plc.process_sim
            # Pack pressure, temperature, conveyor speed, motor RPM, and state
            telemetry_payload = struct.pack(">ffffB", 
                sim.pression, sim.temp_four, sim.vitesse_conv, sim.rpm, sim.machine_state
            )
            
            for sub_id, sub_info in list(plc.subscriptions.items()):
                peer = sub_info["peer"]
                writer = plc.active_writers.get(peer)
                if writer:
                    msg = build_s7plus_message(
                        OPCODE_NOTIFICATION, 
                        FUNC_TRIGGER_VARIABLE_UPDATE, 
                        plc.next_seq(), 
                        0, 
                        telemetry_payload
                    )
                    try:
                        writer.write(msg)
                        await writer.drain()
                    except Exception:
                        pass


# ============================================================================
# HMI CLIENT
# ============================================================================
async def s7plus_hmi_client(port=SERVER_PORT):
    await asyncio.sleep(1.5)
    
    try:
        reader, writer = await asyncio.open_connection(SERVER_IP, port)
    except Exception as e:
        logger.error(f"[HMI] Connection failed: {e}")
        return
    
    logger.info("[HMI] Connected to S7commPlus PLC")
    seq = 0
    
    try:
        # CreateObject
        seq += 1
        msg = build_s7plus_message(OPCODE_REQUEST, FUNC_CREATE_OBJECT, seq, 0)
        writer.write(msg)
        await writer.drain()
        logger.info("[HMI] Sent CreateObject")
        
        tpkt = await reader.readexactly(4)
        length = struct.unpack(">H", tpkt[2:4])[0]
        await reader.readexactly(length - 4)  # FIX: Subtract 4 TPKT header bytes
        
        # CreateSubscription
        seq += 1
        symbol_ids = [SymbolTable.symbol_to_id(name) for name in ["Pression", "TempFour", "VitesseConv"]]
        payload = b"".join(struct.pack(">I", sid) for sid in symbol_ids)
        msg = build_s7plus_message(OPCODE_REQUEST, FUNC_CREATE_SUBSCRIPTION, seq, 0, payload)
        writer.write(msg)
        await writer.drain()
        logger.info(f"[HMI] Sent CreateSubscription for {len(symbol_ids)} symbols")
        
        tpkt = await reader.readexactly(4)
        length = struct.unpack(">H", tpkt[2:4])[0]
        rest = await reader.readexactly(length - 4)  # FIX: Subtract 4 TPKT header bytes
        
        sub_result = parse_s7plus_message(tpkt + rest)
        if sub_result and len(sub_result[4]) >= 2:
            sub_id = struct.unpack(">H", sub_result[4][:2])[0]
            logger.info(f"[HMI] Subscription ID: {sub_id}")
        
        # Listen indefinitely for telemetry notifications
        logger.info("[HMI] Listening for process telemetry...")
        last_state = None  # Added state tracker
        
        while running:
            try:
                tpkt = await asyncio.wait_for(reader.readexactly(4), timeout=3.0)
                length = struct.unpack(">H", tpkt[2:4])[0]
                rest = await reader.readexactly(length - 4)
                result = parse_s7plus_message(tpkt + rest)
                
                if result and result[0] == OPCODE_NOTIFICATION:
                    payload = result[4]
                    if len(payload) >= 17:
                        press, temp, speed, rpm, state = struct.unpack(">ffffB", payload[:17])
                        states = {0: "IDLE", 1: "RUNNING", 2: "PAINTING", 3: "QC", 4: "EJECT"}
                        state_str = states.get(state, "UNKNOWN")
                        
                        if state != last_state:  # Added duplicate filter
                            logger.info(
                                f"[TELEMETRY] State: {state_str:<8} |"
                                f"Pressure: {press:.2f} bar |"
                                f"Temp: {temp:.1f}°C |"
                                f"Speed: {speed:.2f} m/s |"
                                f"RPM: {rpm:.0f}"
                            )
                            last_state = state  # Updated state tracker
            except asyncio.TimeoutError:
                pass
            
    except Exception as e:
        logger.error(f"[HMI] Exception: {e}")
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass
        logger.info("[HMI] Disconnected")


# ============================================================================
# MAIN EXECUTION
# ============================================================================
async def main():
    global running
    
    logger.info("=" * 70)
    logger.info("NATIVE S7COMMPLUS PLC SERVER")
    logger.info("=" * 70)
    
    plc = S7PlusPLC()
    
    try:
        server = await asyncio.start_server(
            lambda r, w: handle_client(r, w, plc), SERVER_IP, SERVER_PORT
        )
        logger.info(f"[SERVER] Listening on {SERVER_IP}:{SERVER_PORT}")
        
        physics_thread = threading.Thread(
            target=background_process_loop, args=(plc,), daemon=True
        )
        physics_thread.start()
        logger.info("[MAIN] Physics thread started\n")
        
        async with server:
            notify_task = asyncio.create_task(notification_sender(plc))
            client_task = asyncio.create_task(s7plus_hmi_client(port=SERVER_PORT))
            
            await client_task
            notify_task.cancel()
            
    except Exception as e:
        logger.error(f"[MAIN] Error: {e}")
    finally:
        running = False
        logger.info("[MAIN] Shutting down...")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass