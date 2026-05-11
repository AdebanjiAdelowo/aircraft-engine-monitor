"""
Level 2 test — virtual UART / ESP32 ACK simulator.

Creates a virtual serial port pair with socat, then listens on one end and
sends ACKs exactly as a real ESP32 would, so rpi_test_final.py can be run
on a Mac or Linux machine without any physical hardware.

Usage (three terminals):

  Terminal 1 — start socat and note the two /dev/ttys??? paths it prints:
      socat -d -d pty,raw,echo=0 pty,raw,echo=0

  Terminal 2 — start this mock, using the SECOND port socat printed:
      python mock_esp32.py /dev/ttys005

  Terminal 3 — edit UART_PORT in rpi_test_final.py to the FIRST socat port,
               then run:
      python rpi_test_final.py
"""

import sys
import struct
import time
import logging

try:
    import serial
except ImportError:
    sys.exit("pyserial not installed — run: pip install pyserial")

# Packet framing constants (must match rpi_test_final.py)
START_BYTE_DATA   = 0xAA
START_BYTE_RECORD = 0xAC
START_BYTE_SYNC   = 0xAB
END_BYTE          = 0x55
BAUDRATE          = 115200

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [ESP32-mock] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def calculate_crc8(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) if (crc & 0x80) else (crc << 1)
            crc &= 0xFF
    return crc


def read_packet(port: serial.Serial):
    """
    Block until a complete framed packet arrives and return
    (start_byte, payload_bytes).  Returns None on timeout.
    """
    start = port.read(1)
    if not start:
        return None

    start_byte = start[0]
    if start_byte not in (START_BYTE_DATA, START_BYTE_RECORD, START_BYTE_SYNC):
        log.warning(f"Unknown start byte 0x{start_byte:02X} — discarding")
        return None

    length_byte = port.read(1)
    if not length_byte:
        return None
    length = length_byte[0]

    payload = port.read(length)
    if len(payload) != length:
        log.warning(f"Short payload: expected {length}, got {len(payload)}")
        return None

    crc_byte = port.read(1)
    end_byte  = port.read(1)

    if not crc_byte or not end_byte:
        return None

    if end_byte[0] != END_BYTE:
        log.warning(f"Bad end byte: 0x{end_byte[0]:02X}")
        return None

    expected_crc = calculate_crc8(payload)
    if crc_byte[0] != expected_crc:
        log.warning(f"CRC mismatch: expected 0x{expected_crc:02X}, got 0x{crc_byte[0]:02X}")
        return None

    return start_byte, payload


def handle_data_packet(port: serial.Serial, payload: bytes, stats: dict):
    """Parse a DATA packet, print values, send ACK."""
    if len(payload) < 15:
        log.warning(f"DATA payload too short: {len(payload)} bytes")
        return

    seq       = struct.unpack_from('<H', payload, 0)[0]
    timestamp = struct.unpack_from('<Q', payload, 2)[0]
    rpm       = struct.unpack_from('<f', payload, 10)[0]
    status    = payload[14]
    freq      = struct.unpack_from('<f', payload, 15)[0] if len(payload) >= 19 else 0.0

    stats['packets_received'] += 1
    log.info(
        f"DATA  seq={seq:5d}  ts={timestamp}ms  "
        f"RPM={rpm:7.1f}  status={'ON' if status else 'OFF'}  freq={freq:.2f}Hz"
    )

    # Send ACK: [0x06, seq_low, seq_high]
    ack = bytes([0x06]) + struct.pack('<H', seq)
    port.write(ack)
    port.flush()


def handle_record_packet(payload: bytes, stats: dict):
    if len(payload) < 8:
        return
    timestamp = struct.unpack_from('<Q', payload, 0)[0]
    stats['record_alerts'] += 1
    log.info(f"RECORD_ALERT  ts={timestamp}ms")


def handle_sync_packet(payload: bytes, stats: dict):
    if len(payload) < 8:
        return
    timestamp = struct.unpack_from('<Q', payload, 0)[0]
    stats['sync_packets'] += 1
    log.info(f"SYNC  ts={timestamp}ms")


def print_stats(stats: dict):
    log.info(
        f"Stats — DATA:{stats['packets_received']}  "
        f"RECORD_ALERT:{stats['record_alerts']}  "
        f"SYNC:{stats['sync_packets']}  "
        f"CRC_errors:{stats['crc_errors']}"
    )


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    port_path = sys.argv[1]
    log.info(f"Opening {port_path} at {BAUDRATE} baud…")

    try:
        port = serial.Serial(port_path, BAUDRATE, timeout=2)
    except serial.SerialException as e:
        sys.exit(f"Could not open {port_path}: {e}")

    log.info("Mock ESP32 ready — waiting for packets (Ctrl+C to stop)")

    stats = {
        'packets_received': 0,
        'record_alerts': 0,
        'sync_packets': 0,
        'crc_errors': 0,
    }

    last_stats_print = time.time()
    STATS_INTERVAL = 30  # seconds

    try:
        while True:
            result = read_packet(port)

            if result is None:
                # Print stats periodically even if no packets arrive
                if time.time() - last_stats_print >= STATS_INTERVAL:
                    print_stats(stats)
                    last_stats_print = time.time()
                continue

            start_byte, payload = result

            if start_byte == START_BYTE_DATA:
                handle_data_packet(port, payload, stats)
            elif start_byte == START_BYTE_RECORD:
                handle_record_packet(payload, stats)
            elif start_byte == START_BYTE_SYNC:
                handle_sync_packet(payload, stats)

            if time.time() - last_stats_print >= STATS_INTERVAL:
                print_stats(stats)
                last_stats_print = time.time()

    except KeyboardInterrupt:
        print()
        log.info("Stopping mock ESP32")
        print_stats(stats)
    finally:
        port.close()


if __name__ == "__main__":
    main()
