"""
=============================================================================
S7COMM INDUSTRIAL PLC SERVER - PRODUCTION-REALISTIC TRAFFIC GENERATION
=============================================================================
Refactored Python-Snap7 S7 PLC server emulation for Siemens S7-1200.
Generates realistic S7comm network traffic for ML/IDS model training.

Features:
  ✓ 256-byte DB1 with strict Big-Endian S7 byte packing
  ✓ Real-time dynamic process simulation (background thread)
  ✓ Continuous mathematical drift (sine/cosine + Gaussian noise)
  ✓ Proper S7 CPU state management
  ✓ Robust Snap7Exception handling
  ✓ Graceful Ctrl+C shutdown with cleanup

DB1 Memory Map (256 bytes):
  Bytes 0-1      : Input Coils (HMI commands)
  Bytes 2-3      : Output Coils (PLC states)
  Bytes 4-7      : Pressure (S7 REAL - 32-bit Float, Big-Endian)
  Bytes 8-11     : Temperature (S7 REAL - 32-bit Float, Big-Endian)
  Bytes 12-15    : Conveyor Speed (S7 REAL - 32-bit Float, Big-Endian)
  Bytes 16-19    : Motor RPM (S7 INT - 16-bit Int, Big-Endian)
  Bytes 20-23    : Flow Rate (S7 REAL - 32-bit Float, Big-Endian)
  Bytes 24-27    : System Uptime (S7 DINT - 32-bit Int, Big-Endian)
  Bytes 28-31    : Cycle Count (S7 DINT - 32-bit Int, Big-Endian)
  Bytes 32-35    : Paint Temperature (S7 REAL, Big-Endian)
  Bytes 36-39    : Paint Pressure (S7 REAL, Big-Endian)
  Bytes 40-255   : Reserved for future telemetry & anomalies
=============================================================================
"""

import asyncio
import random
import struct
import threading
import time
import logging
import numpy as np
from datetime import datetime

try:
    import snap7
except ImportError:
    print("[!] Error: python-snap7 not installed. Install with: pip install python-snap7")
    exit(1)

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("S7PLC")
logger.setLevel(logging.INFO)

# Suppress snap7 debug noise
logging.getLogger("snap7").setLevel(logging.WARNING)

# ============================================================================
# 1. CONFIGURATION
# ============================================================================
SERVER_IP = "127.0.0.1"
SERVER_PORT = 102
DB_NUMBER = 1
DB_SIZE = 256  # 256 bytes (expanded from 20 bytes)

# Physics simulation parameters
PHYSICS_UPDATE_INTERVAL = 0.05  # 50ms (20 Hz sampling)
SINE_FREQUENCY = 0.1            # 0.1 Hz oscillation
ANOMALY_INJECTION_RATE = 0.001  # 0.1% chance per update

# Telemetry baseline values
PRESSURE_BASE = 2.5             # bar
TEMP_BASE = 45.0                # °C
SPEED_BASE = 1.2                # m/min
RPM_BASE = 1450                 # motor RPM
FLOW_RATE_BASE = 15.5           # L/min

# Running flag for graceful shutdown
running = True


# ============================================================================
# 2. PROCESS STATE & PHYSICS SIMULATOR
# ============================================================================
class ProcessSimulator:
    """Realistic industrial process simulator with continuous drift."""
    
    def __init__(self):
        self.lock = threading.Lock()
        self.time_accumulator = 0.0
        self.anomaly_active = False
        self.anomaly_start = 0.0
        self.anomaly_duration = 0.0
        
        # Process state
        self.pressure = PRESSURE_BASE
        self.temperature = TEMP_BASE
        self.speed = SPEED_BASE
        self.rpm = RPM_BASE
        self.flow_rate = FLOW_RATE_BASE
        self.paint_temp = TEMP_BASE + 5.0
        self.paint_pressure = PRESSURE_BASE + 1.5
        
        # Operational state
        self.machine_state = 0  # 0=idle, 1=running, 2=painting, 3=quality_check, 4=eject
        self.cycle_count = 0
        self.uptime_seconds = 0.0
        
    def update(self, dt: float, machine_state: int):
        """Update process variables with realistic dynamics."""
        with self.lock:
            self.time_accumulator += dt
            self.uptime_seconds += dt
            self.machine_state = machine_state
            
            # Inject anomalies at low frequency
            if random.random() < ANOMALY_INJECTION_RATE:
                self._trigger_anomaly()
            
            # Apply anomaly if active
            if self.anomaly_active:
                elapsed = time.time() - self.anomaly_start
                if elapsed > self.anomaly_duration:
                    self.anomaly_active = False
                else:
                    self._apply_anomaly()
            
            # State-dependent baseline
            if machine_state == 0:  # Idle
                base_pressure = PRESSURE_BASE * 0.8
                base_temp = TEMP_BASE
                base_speed = 0.0
                base_rpm = 0.0
                
            elif machine_state == 1:  # Running (conveyor)
                base_pressure = PRESSURE_BASE
                base_temp = TEMP_BASE + random.gauss(0, 2.0)
                base_speed = SPEED_BASE
                base_rpm = RPM_BASE
                
            elif machine_state == 2:  # Painting
                base_pressure = 4.2 + random.gauss(0, 0.2)
                base_temp = 50.0 + random.gauss(0, 1.5)
                base_speed = 0.0
                base_rpm = 0.0
                
            elif machine_state == 3:  # Quality check
                base_pressure = PRESSURE_BASE * 0.9
                base_temp = TEMP_BASE + random.gauss(0, 1.0)
                base_speed = 0.0
                base_rpm = 0.0
                
            elif machine_state == 4:  # Eject
                base_pressure = 5.5 + random.gauss(0, 0.3)
                base_temp = TEMP_BASE + random.gauss(0, 2.0)
                base_speed = 0.0
                base_rpm = 500.0 + random.gauss(0, 50)
            else:
                base_pressure = PRESSURE_BASE
                base_temp = TEMP_BASE
                base_speed = SPEED_BASE
                base_rpm = RPM_BASE
            
            # Apply sine wave drift (realistic sensor oscillation)
            sine_component = 0.3 * np.sin(2 * np.pi * SINE_FREQUENCY * self.time_accumulator)
            
            # Add Gaussian noise for realistic sensor fluctuation
            noise_pressure = random.gauss(0, 0.15)
            noise_temp = random.gauss(0, 1.2)
            noise_speed = random.gauss(0, 0.08)
            noise_rpm = random.gauss(0, 20)
            noise_flow = random.gauss(0, 0.5)
            
            # Update all sensor values
            self.pressure = max(0.5, base_pressure + sine_component * 0.5 + noise_pressure)
            self.temperature = max(20.0, base_temp + sine_component + noise_temp)
            self.speed = max(0.0, base_speed + sine_component * 0.1 + noise_speed)
            self.rpm = max(0.0, base_rpm + sine_component * 100 + noise_rpm)
            self.flow_rate = max(0.0, FLOW_RATE_BASE + sine_component + noise_flow)
            
            # Paint-specific telemetry
            if machine_state == 2:
                self.paint_temp = self.temperature + 5.0
                self.paint_pressure = self.pressure + 1.5
            else:
                self.paint_temp = TEMP_BASE + 3.0
                self.paint_pressure = PRESSURE_BASE + 0.5
    
    def _trigger_anomaly(self):
        """Trigger a realistic anomaly event."""
        anomaly_type = random.choice(['pressure_spike', 'temp_drop', 'speed_loss', 'sensor_noise'])
        self.anomaly_active = True
        self.anomaly_start = time.time()
        self.anomaly_duration = random.uniform(2.0, 8.0)
        logger.warning(f"[ANOMALY] Triggered: {anomaly_type} (duration: {self.anomaly_duration:.1f}s)")
    
    def _apply_anomaly(self):
        """Apply active anomaly to sensor values."""
        if self.anomaly_active:
            elapsed = time.time() - self.anomaly_start
            progress = elapsed / self.anomaly_duration
            
            # Ramping anomaly effect
            if progress < 0.3:
                effect = progress / 0.3  # Ramp up
            elif progress < 0.7:
                effect = 1.0  # Full strength
            else:
                effect = (1.0 - progress) / 0.3  # Ramp down
            
            # Random anomaly effects
            self.pressure += random.gauss(0.5, 0.3) * effect
            self.temperature += random.gauss(3.0, 1.0) * effect
            self.rpm += random.gauss(200, 50) * effect
    
    def get_state(self):
        """Thread-safe state snapshot."""
        with self.lock:
            return {
                'pressure': self.pressure,
                'temperature': self.temperature,
                'speed': self.speed,
                'rpm': self.rpm,
                'flow_rate': self.flow_rate,
                'paint_temp': self.paint_temp,
                'paint_pressure': self.paint_pressure,
                'uptime': self.uptime_seconds,
                'cycle_count': self.cycle_count,
                'machine_state': self.machine_state,
            }


# ============================================================================
# 3. S7 PLC SERVER (VIRTUAL S7-1200)
# ============================================================================
class S7PLCServer:
    """Snap7-based S7 PLC server with proper memory management."""
    
    def __init__(self, ip=SERVER_IP, port=SERVER_PORT):
        self.ip = ip
        self.port = port
        self.server = None
        self.db1_data = bytearray(DB_SIZE)
        self.process_sim = ProcessSimulator()
        self.server_running = False
        self.update_counter = 0
        
    def start(self):
        """Initialize and start the Snap7 server."""
        try:
            self.server = snap7.server.Server()
            
            # Configure server for non-standard port if needed
            if self.port != 102:
                self.server.set_param(snap7.type.Parameter.LocalPort, self.port)
            
            # Register DB1 with 256 bytes
            self.server.register_area(snap7.type.SrvArea.DB, DB_NUMBER, self.db1_data)
            
            # Start the server
            self.server.start_to(self.ip, tcp_port=self.port)
            self.server_running = True
            
            logger.info(f"[SERVER] S7 PLC Server started on {self.ip}:{self.port}")
            logger.info(f"[SERVER] DB{DB_NUMBER} Memory: {DB_SIZE} bytes")
            logger.info(f"[SERVER] CPU Status: RUN")
            
        except Exception as e:
            logger.error(f"[SERVER] Snap7Exception: {e}")
            raise
        except OSError as e:
            if "Address already in use" in str(e):
                logger.error(f"[SERVER] Port {self.port} already in use")
            else:
                logger.error(f"[SERVER] OSError: {e}")
            raise
        except Exception as e:
            logger.error(f"[SERVER] Unexpected error: {e}", exc_info=True)
            raise
    
    def stop(self):
        """Stop the server gracefully."""
        try:
            if self.server and self.server_running:
                self.server.stop()
                self.server_running = False
                logger.info("[SERVER] S7 PLC Server stopped")
        except Exception as e:
            logger.warning(f"[SERVER] Error stopping server: {e}")
    
    def destroy(self):
        """Destroy server resources."""
        try:
            if self.server:
                self.server.destroy()
                logger.info("[SERVER] S7 PLC Server resources destroyed")
        except Exception as e:
            logger.warning(f"[SERVER] Error destroying server: {e}")
    
    def update_db1(self, inputs: dict, outputs: dict):
        """Encode process state into DB1 with strict Big-Endian packing."""
        try:
            # Bytes 0-1: Input coils (HMI commands)
            input_byte = 0
            if inputs.get('start'):
                input_byte |= (1 << 0)
            if inputs.get('stop'):
                input_byte |= (1 << 1)
            if inputs.get('paint_trigger'):
                input_byte |= (1 << 2)
            if inputs.get('quality_check'):
                input_byte |= (1 << 3)
            if inputs.get('quality_ok'):
                input_byte |= (1 << 4)
            self.db1_data[0] = input_byte
            
            # Bytes 2-3: Output coils (PLC states)
            output_byte = 0
            if outputs.get('conveyor'):
                output_byte |= (1 << 0)
            if outputs.get('paint_valve'):
                output_byte |= (1 << 1)
            if outputs.get('eject_valve'):
                output_byte |= (1 << 2)
            self.db1_data[2] = output_byte
            
            # Get current process state
            proc_state = self.process_sim.get_state()
            
            # Bytes 4-7: Pressure (S7 REAL - 32-bit Float, Big-Endian)
            struct.pack_into(">f", self.db1_data, 4, proc_state['pressure'])
            
            # Bytes 8-11: Temperature (S7 REAL, Big-Endian)
            struct.pack_into(">f", self.db1_data, 8, proc_state['temperature'])
            
            # Bytes 12-15: Conveyor Speed (S7 REAL, Big-Endian)
            struct.pack_into(">f", self.db1_data, 12, proc_state['speed'])
            
            # Bytes 16-19: Motor RPM (S7 INT, Big-Endian - 16-bit)
            struct.pack_into(">h", self.db1_data, 16, int(proc_state['rpm']))
            
            # Bytes 20-23: Flow Rate (S7 REAL, Big-Endian)
            struct.pack_into(">f", self.db1_data, 20, proc_state['flow_rate'])
            
            # Bytes 24-27: System Uptime (S7 DINT, Big-Endian - 32-bit)
            struct.pack_into(">i", self.db1_data, 24, int(proc_state['uptime']))
            
            # Bytes 28-31: Cycle Count (S7 DINT, Big-Endian - 32-bit)
            struct.pack_into(">i", self.db1_data, 28, proc_state['cycle_count'])
            
            # Bytes 32-35: Paint Temperature (S7 REAL, Big-Endian)
            struct.pack_into(">f", self.db1_data, 32, proc_state['paint_temp'])
            
            # Bytes 36-39: Paint Pressure (S7 REAL, Big-Endian)
            struct.pack_into(">f", self.db1_data, 36, proc_state['paint_pressure'])
            
            # Log periodic updates
            self.update_counter += 1
            if self.update_counter % 200 == 0:  # Every 10 seconds at 20Hz
                logger.debug(
                    f"[DB1] P={proc_state['pressure']:.2f}bar "
                    f"T={proc_state['temperature']:.1f}°C "
                    f"S={proc_state['speed']:.2f}m/min "
                    f"RPM={proc_state['rpm']:.0f} "
                    f"State={proc_state['machine_state']}"
                )
        
        except struct.error as e:
            logger.error(f"[DB1] Struct packing error: {e}")
        except Exception as e:
            logger.error(f"[DB1] Unexpected error updating DB1: {e}", exc_info=True)
    
    def read_inputs(self):
        """Read input coils from DB1."""
        try:
            input_byte = self.db1_data[0]
            return {
                'start': bool(input_byte & (1 << 0)),
                'stop': bool(input_byte & (1 << 1)),
                'paint_trigger': bool(input_byte & (1 << 2)),
                'quality_check': bool(input_byte & (1 << 3)),
                'quality_ok': bool(input_byte & (1 << 4)),
            }
        except Exception as e:
            logger.error(f"[DB1] Error reading inputs: {e}")
            return {}


# ============================================================================
# 4. BACKGROUND PROCESS SIMULATOR THREAD
# ============================================================================
def background_process_loop(plc_server: S7PLCServer):
    """Dedicated thread for continuous process simulation and DB1 updates."""
    logger.info("[PHYSICS] Background process simulation started")
    
    try:
        last_update = time.time()
        machine_state = 0
        
        while running:
            try:
                now = time.time()
                dt = now - last_update
                last_update = now
                
                # Read current HMI inputs
                inputs = plc_server.read_inputs()
                
                # Determine machine state based on inputs
                if inputs.get('start') and not inputs.get('stop'):
                    machine_state = 1  # Running
                    if inputs.get('paint_trigger'):
                        machine_state = 2  # Painting
                    elif inputs.get('quality_check'):
                        machine_state = 3  # Quality check
                        if inputs.get('quality_ok'):
                            machine_state = 3  # Still in QC
                        else:
                            machine_state = 4  # Eject
                else:
                    machine_state = 0  # Idle
                
                # Update process simulator
                plc_server.process_sim.update(dt, machine_state)
                
                # Determine PLC outputs
                outputs = {
                    'conveyor': machine_state == 1,
                    'paint_valve': machine_state == 2,
                    'eject_valve': machine_state == 4,
                }
                
                # Update DB1 with latest process state
                plc_server.update_db1(inputs, outputs)
                
                # Sleep for physics update interval
                time.sleep(PHYSICS_UPDATE_INTERVAL)
            
            except snap7.Snap7Exception as e:
                logger.error(f"[PHYSICS] Snap7Exception: {e}")
                time.sleep(0.1)  # Brief sleep before retry
            except Exception as e:
                logger.error(f"[PHYSICS] Error in process loop: {e}", exc_info=False)
                time.sleep(0.1)
    
    except KeyboardInterrupt:
        logger.info("[PHYSICS] Process loop interrupted")
    except Exception as e:
        logger.error(f"[PHYSICS] Unexpected error: {e}", exc_info=True)
    finally:
        logger.info("[PHYSICS] Background process simulation stopped")


# ============================================================================
# 5. HMI CLIENT (AUTONOMOUS S7COMM TRAFFIC)
# ============================================================================
async def hmi_client_loop(port=SERVER_PORT):
    """Simulates HMI client with realistic S7comm traffic patterns."""
    await asyncio.sleep(2)  # Wait for server to start
    
    client = snap7.client.Client()
    
    def ensure_connected():
        """Helper to ensure socket connection is alive, reconnecting if needed."""
        if not client.get_connected():
            logger.warning("[HMI] Client disconnected or session dropped. Attempting reconnect...")
            try:
                client.disconnect()
            except Exception:
                pass
            client.connect(SERVER_IP, 0, 1, tcp_port=port)
            logger.info("[HMI] Reconnected to S7 PLC server")

    # Initial connection attempt
    try:
        ensure_connected()
    except (ConnectionRefusedError, OSError) as e:
        logger.error(f"[HMI] Connection failed (server not running): {e}")
        return
    except Exception as e:
        logger.error(f"[HMI] Unexpected connection error: {e}")
        return
    
    cycle = 0
    
    try:
        while running:
            cycle += 1
            logger.info(f"\n{'='*70}")
            logger.info(f"PRODUCTION CYCLE #{cycle}")
            logger.info(f"{'='*70}")
            
            try:
                # Ensure connection is active prior to cycle execution
                ensure_connected()

                # START command
                logger.info("[HMI] ► Sending START command")
                input_data = bytearray(1)
                input_data[0] |= (1 << 0)  # Set start bit
                client.db_write(DB_NUMBER, 0, input_data)
                await asyncio.sleep(0.2)
                
                # Clear start bit
                input_data[0] &= ~(1 << 0)
                client.db_write(DB_NUMBER, 0, input_data)
                logger.info("[HMI] ✓ START sent")
                
                # Read sensor telemetry
                await asyncio.sleep(0.5)
                try:
                    data = client.db_read(DB_NUMBER, 0, 40)
                    if data and len(data) >= 40:
                        pressure = struct.unpack(">f", data[4:8])[0]
                        temp = struct.unpack(">f", data[8:12])[0]
                        speed = struct.unpack(">f", data[12:16])[0]
                        logger.info(
                            f"[HMI] ✓ Telemetry: P={pressure:.2f}bar "
                            f"T={temp:.1f}°C S={speed:.2f}m/min"
                        )
                except Exception as e:
                    logger.warning(f"[HMI] Failed to read telemetry: {e}")
                
                # PAINT trigger
                logger.info("[HMI] ► Triggering PAINT command")
                input_data[0] |= (1 << 2)  # Set paint trigger
                client.db_write(DB_NUMBER, 0, input_data)
                await asyncio.sleep(0.5)
                input_data[0] &= ~(1 << 2)
                client.db_write(DB_NUMBER, 0, input_data)
                logger.info("[HMI] ✓ PAINT triggered")
                
                # Simulate painting with realistic full-scale duration
                paint_duration = random.uniform(60.0, 80.0)
                logger.info(f"[HMI] ► Painting in progress (~{paint_duration:.0f}s)")
                await asyncio.sleep(paint_duration)
                
                # QUALITY CHECK
                logger.info("[HMI] ► Initiating QUALITY CHECK")
                input_data[0] |= (1 << 3)  # Set QC bit
                is_ok = random.random() < 0.85
                if is_ok:
                    input_data[0] |= (1 << 4)  # Set OK bit
                else:
                    input_data[0] &= ~(1 << 4)  # Clear OK bit
                
                client.db_write(DB_NUMBER, 0, input_data)
                await asyncio.sleep(1.0)
                
                if is_ok:
                    logger.info("[HMI] ✓ Quality: PASS")
                else:
                    logger.warning("[HMI] ⚠ Quality: FAIL - Defect detected")
                    await asyncio.sleep(3.0)
                
                # Clear QC/OK bits
                input_data[0] &= ~((1 << 3) | (1 << 4))
                client.db_write(DB_NUMBER, 0, input_data)
                
                # STOP
                logger.info("[HMI] ► Sending STOP command")
                input_data[0] |= (1 << 1)  # Set stop bit
                client.db_write(DB_NUMBER, 0, input_data)
                await asyncio.sleep(0.2)
                input_data[0] &= ~((1 << 0) | (1 << 1))  # Clear start/stop
                client.db_write(DB_NUMBER, 0, input_data)
                logger.info("[HMI] ✓ Cycle completed\n")
                
                await asyncio.sleep(1.0)
            
            except (ConnectionResetError, OSError, snap7.Snap7Exception) as e:
                logger.error(f"[HMI] Network/Snap7 error during cycle #{cycle}: {e}")
                await asyncio.sleep(2.0)  # Brief pause before next cycle attempts auto-reconnect
            except Exception as e:
                logger.error(f"[HMI] Error in cycle #{cycle}: {e}", exc_info=False)
                await asyncio.sleep(1.0)
    
    except KeyboardInterrupt:
        logger.info("[HMI] Client interrupted")
    except Exception as e:
        logger.error(f"[HMI] Unexpected error: {e}", exc_info=True)
    finally:
        try:
            client.disconnect()
            logger.info("[HMI] Client disconnected")
        except Exception as e:
            logger.warning(f"[HMI] Error disconnecting: {e}")


# ============================================================================
# 6. MAIN ENTRY POINT
# ============================================================================
async def main():
    """Main async entry point with proper error handling."""
    global running
    
    logger.info("="*70)
    logger.info("S7COMM INDUSTRIAL PLC SERVER - PRODUCTION-REALISTIC TRAFFIC")
    logger.info("="*70)
    logger.info("Features:")
    logger.info("  ✓ 256-byte DB1 with Big-Endian S7 byte packing")
    logger.info("  ✓ Real-time dynamic process simulation (background thread)")
    logger.info("  ✓ Continuous mathematical drift (sine/cosine + Gaussian noise)")
    logger.info("  ✓ Realistic anomaly injection (0.1% occurrence rate)")
    logger.info("  ✓ Robust Snap7Exception handling")
    logger.info("  ✓ Graceful Ctrl+C shutdown with cleanup")
    logger.info("="*70 + "\n")
    
    plc = None
    physics_thread = None
    
    try:
        # Initialize and start PLC server
        plc = S7PLCServer(ip=SERVER_IP, port=SERVER_PORT)
        plc.start()
        
        # Start background process simulation thread
        physics_thread = threading.Thread(
            target=background_process_loop,
            args=(plc,),
            daemon=True,
            name="ProcessSimulator"
        )
        physics_thread.start()
        logger.info("[MAIN] Process simulation thread started\n")
        
        # Run HMI client (blocks until exception or Ctrl+C)
        await hmi_client_loop(port=SERVER_PORT)
    
    except snap7.Snap7Exception as e:
        logger.error(f"[MAIN] Snap7Exception: {e}")
    except KeyboardInterrupt:
        logger.info("\n[MAIN] Keyboard interrupt (Ctrl+C) received")
    except OSError as e:
        logger.error(f"[MAIN] OSError: {e}")
    except Exception as e:
        logger.error(f"[MAIN] Unexpected error: {e}", exc_info=True)
    finally:
        # Graceful shutdown sequence
        logger.info("[MAIN] Initiating graceful shutdown...")
        running = False
        
        # Stop server
        if plc:
            try:
                plc.stop()
                plc.destroy()
                logger.info("[MAIN] S7 PLC Server stopped and cleaned up")
            except Exception as e:
                logger.warning(f"[MAIN] Error during server cleanup: {e}")
        
        # Wait for physics thread to finish
        if physics_thread and physics_thread.is_alive():
            physics_thread.join(timeout=2.0)
            logger.info("[MAIN] Process simulation thread terminated")
        
        logger.info("="*70)
        logger.info("S7COMM PLC SERVER STOPPED")
        logger.info("="*70)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"[STARTUP] Fatal error: {e}", exc_info=True)