import time
import random
import threading
import asyncio
import sys
import logging
import warnings
from datetime import datetime
from dataclasses import dataclass
import numpy as np

# Suppress pymodbus deprecation warnings (using v4.0+ API correctly)
warnings.filterwarnings("ignore", message=".*ModbusDeviceContext.*deprecated.*")
warnings.filterwarnings("ignore", message=".*ModbusSequentialDataBlock.*deprecated.*")
warnings.filterwarnings("ignore", message=".*ModbusServerContext.*deprecated.*")

from pymodbus.server import StartTcpServer
from pymodbus.datastore import (
    ModbusSequentialDataBlock,
    ModbusServerContext,
    ModbusDeviceContext,
)
from pymodbus.client import ModbusTcpClient


# ============================================================
# LOGGING CONFIGURATION
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Reduce noise from pymodbus logging
logging.getLogger("pymodbus").setLevel(logging.WARNING)


# ============================================================
# CONFIGURATION
# ============================================================
HOST = "127.0.0.1"
PORT = 502
UNIT_ID = 1

# Physics simulation parameters
PHYSICS_UPDATE_INTERVAL = 0.1  # seconds
JUMP_EVENT_INTERVAL = 300  # cycles (~30 seconds at 0.1s interval)
PRESSURE_BASE = 0.53
TEMP_BASE = 27.5
SPEED_BASE = 1.35

# ============================================================
# MODBUS ADDRESS MAP
# ============================================================
# Coils (Outputs / State)
COIL_CONVEYOR = 0       # M_Convoyeur
COIL_PAINTING = 1       # V_Peinture
COIL_EJECT = 2          # V_Eject

# Coils (Inputs / Control Commands)
COIL_START = 8          # HMI_Start
COIL_STOP = 9           # HMI_Stop
COIL_S_PAINT = 10       # HMI_S_Peinture
COIL_S_QC = 11          # HMI_S_QC
COIL_S_OK = 12          # HMI_S_OK

# Holding Registers (FC 03) - Scaled to int16
HR_PAINTING_TIME = 0      # Painting duration in ms (scaled)
HR_EJECT_TIME = 1         # Eject duration in ms (scaled)
HR_PRESSURE = 2           # Pressure in bar * 100
HR_TEMPERATURE = 3        # Temperature in °C * 100
HR_SPEED = 4              # Speed in m/min * 100
HR_CYCLE_COUNT = 5        # Production cycle counter
HR_ERROR_CODE = 6         # Error status code

# Input Registers (FC 04) - Read-only sensor data (scaled)
IR_PRESSURE = 10          # Real-time pressure
IR_TEMPERATURE = 11       # Real-time temperature
IR_SPEED = 12             # Real-time conveyor speed
IR_UPTIME = 13            # System uptime in seconds
IR_STATE = 14             # Machine state (0=idle, 1=running, 2=painting, 3=ejecting)

running = True


# ============================================================
# PHYSICS STATE & SIMULATION ENGINE
# ============================================================
@dataclass
class PhysicsState:
    """Encapsulates physics simulation state"""
    cycle_count: int = 0
    pressure: float = PRESSURE_BASE
    temperature: float = TEMP_BASE
    speed: float = SPEED_BASE
    uptime_seconds: float = 0.0
    machine_state: int = 0  # 0=idle, 1=running, 2=painting, 3=ejecting
    
    # Tracking for sine wave
    time_accumulator: float = 0.0
    sine_phase: float = 0.0


class PhysicsSimulator:
    """Async physics simulator with realistic sensor dynamics"""
    
    def __init__(self, state: PhysicsState):
        self.state = state
        self.lock = threading.Lock()
        self.last_jump_cycle = 0
    
    def update(self, dt: float, machine_state: int):
        """
        Update physics state with realistic sensor dynamics.
        
        Args:
            dt: Delta time since last update (seconds)
            machine_state: Current machine operational state
        """
        with self.lock:
            self.state.cycle_count += 1
            self.state.uptime_seconds += dt
            self.state.machine_state = machine_state
            self.state.time_accumulator += dt
            
            # Increment phase for sine wave
            self.state.sine_phase = (self.state.sine_phase + dt * 0.5) % (2 * np.pi)
            
            # Base sine wave + Gaussian noise
            sine_component = 0.2 * np.sin(self.state.sine_phase)
            pressure_noise = random.gauss(0, 0.08)
            temp_noise = random.gauss(0, 0.5)
            speed_noise = random.gauss(0, 0.05)
            
            # State-dependent physics
            if machine_state == 0:  # Idle
                self.state.pressure = PRESSURE_BASE + sine_component + pressure_noise
                self.state.temperature = TEMP_BASE + sine_component * 2 + temp_noise
                self.state.speed = 0.0
                
            elif machine_state == 1:  # Running (conveyor)
                self.state.pressure = PRESSURE_BASE + sine_component + pressure_noise
                self.state.temperature = TEMP_BASE + sine_component * 2 + temp_noise
                self.state.speed = SPEED_BASE + sine_component * 0.15 + speed_noise
                
            elif machine_state == 2:  # Painting
                self.state.pressure = 4.40 + random.gauss(0, 0.1)
                self.state.temperature = 23.6 + random.gauss(0, 0.5)
                self.state.speed = 0.0
                
            elif machine_state == 3:  # Ejecting
                self.state.pressure = 5.0 + random.gauss(0, 0.3)
                self.state.temperature = 33.0 + random.gauss(0, 2.0)
                self.state.speed = 0.0
            
            # Periodic jump events (e.g., cooling fan activation)
            if (self.state.cycle_count - self.last_jump_cycle) >= JUMP_EVENT_INTERVAL:
                self._inject_jump_event()
                self.last_jump_cycle = self.state.cycle_count
    
    def _inject_jump_event(self):
        """Simulate sudden operational changes (e.g., fan kick-in, setpoint shift)"""
        event_type = random.choice(['cooling', 'pressure_spike', 'speed_ramp'])
        
        if event_type == 'cooling':
            self.state.temperature -= random.uniform(2.0, 4.0)
            logger.info(f"[PHYSICS] Jump Event: Cooling fan activated (temp -> {self.state.temperature:.1f}°C)")
        
        elif event_type == 'pressure_spike':
            self.state.pressure += random.uniform(0.5, 1.2)
            logger.info(f"[PHYSICS] Jump Event: Pressure spike detected (pressure -> {self.state.pressure:.2f} bar)")
        
        elif event_type == 'speed_ramp':
            self.state.speed += random.uniform(0.1, 0.3)
            logger.info(f"[PHYSICS] Jump Event: Speed increase (speed -> {self.state.speed:.2f} m/min)")
    
    def get_state(self) -> PhysicsState:
        """Thread-safe state read"""
        with self.lock:
            return PhysicsState(**self.state.__dict__)


# ============================================================
# DATASTORE SETUP - Modern PyModbus v4.0+ Compatible
# ============================================================
def create_context():
    """Create thread-safe datastore for PyModbus 4.0+"""
    # Initialize data blocks with 100 addresses each
    coils = ModbusSequentialDataBlock(1, [False] * 100)
    discrete_inputs = ModbusSequentialDataBlock(1, [False] * 100)
    holding_registers = ModbusSequentialDataBlock(1, [0] * 100)
    input_registers = ModbusSequentialDataBlock(1, [0] * 100)
    
    device = ModbusDeviceContext(
        co=coils,
        di=discrete_inputs,
        hr=holding_registers,
        ir=input_registers
    )
    context = ModbusServerContext(devices={UNIT_ID: device}, single=False)
    
    logger.info("[DATASTORE] Modbus context initialized with 100 addresses each (v4.0+ compatible)")
    return context, device  # Return both context and device reference


context, device = create_context()
physics_state = PhysicsState()
physics_sim = PhysicsSimulator(physics_state)


# ============================================================
# REGISTER ACCESS HELPERS - THREAD-SAFE
# ============================================================
def get_coil(address):
    """Read a coil value (FC 01/02) - PyModbus v4.0+ Compatible"""
    try:
        result = context.getValues(1, address, count=1, unit=UNIT_ID)
        return bool(result[0]) if result else False
    except Exception as e:
        logger.debug(f"[COIL_READ] Read coil {address}: {e}")
        return False


def set_coil(address, value):
    """Write a coil value (FC 05/15) - PyModbus v4.0+ Compatible"""
    try:
        context.setValues(1, address, [bool(value)], unit=UNIT_ID)
        return True
    except Exception as e:
        logger.debug(f"[COIL_WRITE] Failed to write coil {address}: {e}")
        return False


def get_holding(address):
    """Read a holding register (FC 03) - PyModbus v4.0+ Compatible"""
    try:
        result = context.getValues(3, address, count=1, unit=UNIT_ID)
        return int(result[0]) if result else 0
    except Exception as e:
        logger.debug(f"[HR_READ] Read holding register {address}: {e}")
        return 0


def set_holding(address, value):
    """Write a holding register (FC 16) - PyModbus v4.0+ Compatible"""
    try:
        context.setValues(3, address, [int(value)], unit=UNIT_ID)
        return True
    except Exception as e:
        logger.debug(f"[HR_WRITE] Failed to write HR {address}: {e}")
        return False


def get_input_register(address):
    """Read an input register (FC 04) - PyModbus v4.0+ Compatible"""
    try:
        result = context.getValues(4, address, count=1, unit=UNIT_ID)
        return int(result[0]) if result else 0
    except Exception as e:
        logger.debug(f"[IR_READ] Read input register {address}: {e}")
        return 0


def set_input_register(address, value):
    """Write an input register (FC 04) - PyModbus v4.0+ Compatible"""
    try:
        context.setValues(4, address, [int(value)], unit=UNIT_ID)
        return True
    except Exception as e:
        logger.debug(f"[IR_WRITE] Write input register {address}: {e}")
        return False


# ============================================================
# ASYNC PHYSICS UPDATE LOOP
# ============================================================
async def async_physics_loop():
    """Background task: continuously update physics and sensor registers"""
    logger.info("[PHYSICS_LOOP] Starting async physics simulation thread")
    
    try:
        while running:
            # Read current machine state from coils
            line_active = get_coil(COIL_START) and not get_coil(COIL_STOP)
            painting_active = get_coil(COIL_PAINTING)
            eject_active = get_coil(COIL_EJECT)
            
            # Determine machine state
            if eject_active:
                machine_state = 3
            elif painting_active:
                machine_state = 2
            elif line_active:
                machine_state = 1
            else:
                machine_state = 0
            
            # Update physics
            physics_sim.update(PHYSICS_UPDATE_INTERVAL, machine_state)
            state = physics_sim.get_state()
            
            # Write sensor data to input registers (scaled to int16)
            set_input_register(IR_PRESSURE, int(state.pressure * 100))
            set_input_register(IR_TEMPERATURE, int(state.temperature * 100))
            set_input_register(IR_SPEED, int(state.speed * 100))
            set_input_register(IR_UPTIME, int(state.uptime_seconds))
            set_input_register(IR_STATE, machine_state)
            
            # Log periodic updates
            if state.cycle_count % 100 == 0:
                logger.debug(
                    f"[PHYSICS] Cycle {state.cycle_count} | "
                    f"P={state.pressure:.2f}bar T={state.temperature:.1f}°C "
                    f"S={state.speed:.2f}m/min State={machine_state}"
                )
            
            await asyncio.sleep(PHYSICS_UPDATE_INTERVAL)
    
    except Exception as e:
        logger.error(f"[PHYSICS_LOOP] Unexpected error: {e}", exc_info=True)
    finally:
        logger.info("[PHYSICS_LOOP] Physics loop terminated")


# ============================================================
# PLC PROCESS LOGIC
# ============================================================
def plc_process():
    """Main PLC logic - runs in separate thread"""
    line_active = False
    painting_timer = 0.0
    painting_duration = 0.0
    eject_timer = 0.0
    eject_duration = 0.0
    
    previous_start = False
    previous_stop = False
    previous_paint = False
    previous_qc = False
    
    logger.info("[PLC] Starting PLC logic thread...")
    time.sleep(1)
    
    try:
        while running:
            now = time.monotonic()
            
            # Read input coils
            start = get_coil(COIL_START)
            stop = get_coil(COIL_STOP)
            s_paint = get_coil(COIL_S_PAINT)
            s_qc = get_coil(COIL_S_QC)
            s_ok = get_coil(COIL_S_OK)
            
            # Latch Logic for Start/Stop
            if start and not previous_start:
                line_active = True
                painting_timer = 0.0
                painting_duration = 0.0
                eject_timer = 0.0
                eject_duration = 0.0
                logger.info("[PLC] Command: START received")
            
            if stop and not previous_stop:
                line_active = False
                painting_timer = 0.0
                painting_duration = 0.0
                eject_timer = 0.0
                eject_duration = 0.0
                logger.info("[PLC] Command: STOP received")
            
            # Painting Process Timing Logic
            if s_paint and not previous_paint:
                painting_timer = now
                painting_duration = random.uniform(60.0, 80.0)
                logger.info(f"[PLC] Process: Painting sequence initiated (target {painting_duration:.2f}s)")
            
            painting_active = False
            if painting_timer > 0:
                elapsed = now - painting_timer
                if elapsed < painting_duration:
                    painting_active = True
                else:
                    actual_ms = int(painting_duration * 1000)
                    set_holding(HR_PAINTING_TIME, actual_ms)
                    painting_timer = 0.0
                    painting_duration = random.uniform(60.0, 80.0)
                    logger.info(f"[PLC] Painting finished successfully ({actual_ms}ms)")
            
            # Ejection / Quality Control Timing Logic
            if s_qc and not previous_qc and not s_ok:
                eject_timer = now
                eject_duration = random.uniform(23.0, 28.0)
                logger.info(f"[PLC] Process: Defect detected, ejection active (target {eject_duration:.2f}s)")
            
            eject_active = False
            if eject_timer > 0:
                elapsed_e = now - eject_timer
                if elapsed_e < eject_duration:
                    eject_active = True
                else:
                    actual_ms = int(eject_duration * 1000)
                    set_holding(HR_EJECT_TIME, actual_ms)
                    eject_timer = 0.0
                    eject_duration = 0.0
                    logger.info(f"[PLC] Ejection completed ({actual_ms}ms)")
            
            # Output Logic Evaluation
            if line_active and not painting_active and not eject_active:
                conveyor = True
                painting_valve = False
                eject_valve = False
            elif painting_active:
                conveyor = False
                painting_valve = True
                eject_valve = False
            elif eject_active:
                conveyor = False
                painting_valve = False
                eject_valve = True
            else:
                conveyor = False
                painting_valve = False
                eject_valve = False
            
            # Write output coils
            set_coil(COIL_CONVEYOR, conveyor and line_active)
            set_coil(COIL_PAINTING, painting_valve)
            set_coil(COIL_EJECT, eject_valve)
            
            # Update cycle counter
            cycle_count = get_holding(HR_CYCLE_COUNT)
            set_holding(HR_CYCLE_COUNT, cycle_count + 1)
            
            previous_start = start
            previous_stop = stop
            previous_paint = s_paint
            previous_qc = s_qc
            
            time.sleep(0.05)
    
    except Exception as e:
        logger.error(f"[PLC] Critical error: {e}", exc_info=True)
        set_holding(HR_ERROR_CODE, 1)
    finally:
        logger.info("[PLC] PLC logic thread terminated")


# ============================================================
# MODBUS SERVER - Thread-Safe Implementation
# ============================================================
def run_server():
    """Run Modbus TCP Server with proper error handling"""
    logger.info(f"[SERVER] Starting Modbus TCP Server on {HOST}:{PORT}")
    server_running = True
    
    try:
        # Start the server (blocking call)
        StartTcpServer(context=context, address=(HOST, PORT))
    except OSError as e:
        if "Address already in use" in str(e):
            logger.error(f"[SERVER] Port {PORT} already in use. Is another instance running?")
        else:
            logger.error(f"[SERVER] Address/port error: {e}")
        server_running = False
    except Exception as e:
        logger.error(f"[SERVER] Unexpected error: {e}", exc_info=True)
        server_running = False
    finally:
        if server_running:
            logger.info("[SERVER] Modbus TCP Server stopped")
        else:
            logger.warning("[SERVER] Modbus TCP Server failed to start")


# ============================================================
# HMI CLIENT - Production-Ready Implementation
# ============================================================
def run_hmi():
    """Run HMI client - sends commands and reads data with robust error handling"""
    # Wait for server to fully initialize
    time.sleep(2.5)
    
    client = ModbusTcpClient(HOST, port=PORT, timeout=2)
    
    # Retry connection logic
    max_retries = 3
    for attempt in range(max_retries):
        if client.connect():
            logger.info("[HMI] Connected to Modbus PLC server")
            break
        else:
            logger.warning(f"[HMI] Connection attempt {attempt + 1}/{max_retries} failed, retrying...")
            time.sleep(0.5)
    else:
        logger.error("[HMI] Failed to connect to Modbus server after 3 attempts")
        return
    
    cycle = 0
    
    try:
        while running:
            cycle += 1
            logger.info(f"\n{'='*70}")
            logger.info(f"PRODUCTION CYCLE #{cycle}")
            logger.info(f"{'='*70}")
            
            try:
                # 1. START pulse
                if not client.write_coil(COIL_START, True):
                    logger.error("[HMI] Failed to send START command (write failed)")
                    continue
                    
                time.sleep(0.5)
                
                if not client.write_coil(COIL_START, False):
                    logger.error("[HMI] Failed to clear START command")
                    continue
                    
                time.sleep(0.2)
                logger.info("[HMI] ✓ START command sent successfully")
                
                # 2. Conveyor approach - read sensor data
                time.sleep(0.5)
                res = client.read_input_registers(IR_PRESSURE, count=5)
                
                if res and not res.isError() and res.registers:
                    pressure = res.registers[0] / 100.0 if res.registers else 0.0
                    temp = res.registers[1] / 100.0 if len(res.registers) > 1 else 0.0
                    speed = res.registers[2] / 100.0 if len(res.registers) > 2 else 0.0
                    logger.info(f"[HMI] ✓ Sensors read: P={pressure:.2f}bar T={temp:.1f}°C V={speed:.2f}m/min")
                else:
                    logger.warning("[HMI] ⚠ Failed to read sensor registers")
                
                # 3. Paint trigger
                logger.info("[HMI] Vehicle detected at Paint Station")
                if not client.write_coil(COIL_S_PAINT, True):
                    logger.error("[HMI] Failed to trigger paint command")
                    continue
                    
                time.sleep(1.0)
                
                paint_duration = random.uniform(60.0, 80.0)
                logger.info(f"[HMI] ► Painting active (simulated {paint_duration:.1f}s)...")
                time.sleep(min(paint_duration, 10.0))  # Truncate for demo
                
                # 4. Stop painting
                if not client.write_coil(COIL_S_PAINT, False):
                    logger.error("[HMI] Failed to stop paint command")
                    continue
                    
                logger.info("[HMI] ✓ Painting completed")
                time.sleep(0.5)
                
                # 5. Quality check
                logger.info("[HMI] Vehicle detected at Quality Control")
                is_ok = random.random() < 0.85
                
                if not client.write_coil(COIL_S_QC, True):
                    logger.error("[HMI] Failed to send QC command")
                    continue
                    
                if not client.write_coil(COIL_S_OK, is_ok):
                    logger.error("[HMI] Failed to send QC result")
                    continue
                
                if is_ok:
                    logger.info("[HMI] ✓ Quality Check: PASS")
                else:
                    logger.warning("[HMI] ⚠ Quality Check: DEFECT (Ejection triggered)")
                    time.sleep(3.0)
                
                if not client.write_coil(COIL_S_QC, False):
                    logger.error("[HMI] Failed to clear QC command")
                    
                if not client.write_coil(COIL_S_OK, False):
                    logger.error("[HMI] Failed to clear QC result")
                
                time.sleep(1.5)
                logger.info("[HMI] ✓ Cycle completed successfully")
            
            except Exception as e:
                logger.error(f"[HMI] Cycle {cycle} failed: {e}", exc_info=False)
                time.sleep(2.0)
                logger.info("[HMI] Attempting to recover...")
    
    except KeyboardInterrupt:
        logger.info("[HMI] Interrupted by user")
    except Exception as e:
        logger.error(f"[HMI] Critical error: {e}", exc_info=True)
    finally:
        try:
            client.close()
            logger.info("[HMI] Client connection closed")
        except Exception as e:
            logger.error(f"[HMI] Error closing client: {e}")


# ============================================================
# MAIN ENTRY POINT - Production-Ready Startup
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("INDUSTRIAL MODBUS SIMULATOR - ICS/SCADA ML DATASET GENERATION")
    print("Production-Ready with Physics Simulation & Concurrent Access")
    print("="*80 + "\n")
    
    logger.info("="*70)
    logger.info("CONFIGURATION SUMMARY")
    logger.info("="*70)
    logger.info(f"  Server Address: {HOST}:{PORT}")
    logger.info(f"  Unit ID: {UNIT_ID}")
    logger.info(f"  Physics Update Interval: {PHYSICS_UPDATE_INTERVAL}s")
    logger.info(f"  Jump Event Interval: {JUMP_EVENT_INTERVAL} cycles (~{JUMP_EVENT_INTERVAL * PHYSICS_UPDATE_INTERVAL}s)")
    logger.info(f"  Baseline Pressure: {PRESSURE_BASE} bar")
    logger.info(f"  Baseline Temperature: {TEMP_BASE}°C")
    logger.info(f"  Baseline Speed: {SPEED_BASE} m/min")
    logger.info("="*70 + "\n")
    
    # Start server thread
    logger.info("[STARTUP] Launching Modbus TCP Server thread...")
    server_thread = threading.Thread(target=run_server, daemon=True, name="ModbusServer")
    server_thread.start()
    
    # Start PLC logic thread
    logger.info("[STARTUP] Launching PLC logic thread...")
    plc_thread = threading.Thread(target=plc_process, daemon=True, name="PLCLogic")
    plc_thread.start()
    
    # Start async physics loop in a separate event loop
    logger.info("[STARTUP] Launching physics simulation thread...")
    physics_loop_thread = threading.Thread(
        target=lambda: asyncio.run(async_physics_loop()),
        daemon=True,
        name="PhysicsLoop"
    )
    physics_loop_thread.start()
    
    logger.info("[STARTUP] All worker threads started successfully")
    logger.info(f"[STARTUP] Modbus TCP Server listening on {HOST}:{PORT}")
    logger.info("[STARTUP] Waiting for HMI client to begin production cycles...\n")
    
    # Run HMI (blocks until Ctrl+C)
    try:
        run_hmi()
    except KeyboardInterrupt:
        logger.info("\n[MAIN] Shutdown signal received (Ctrl+C)")
    except Exception as e:
        logger.error(f"[MAIN] Fatal error: {e}", exc_info=True)
    finally:
        logger.info("[MAIN] Initiating graceful shutdown...")
        running = False
        time.sleep(1)
        
        logger.info("[MAIN] Waiting for threads to terminate...")
        server_thread.join(timeout=2)
        plc_thread.join(timeout=2)
        physics_loop_thread.join(timeout=2)
        
        logger.info("="*70)
        logger.info("SIMULATION ENVIRONMENT STOPPED")
        logger.info("="*70 + "\n")
        sys.exit(0)