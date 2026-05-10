from enum import Enum
import zmq
import struct

class RequestType(Enum):
    START_EXPERIMENT = 1
    STOP_EXPERIMENT = 2
    START_RECORDING = 3
    STOP_RECORDING = 4
    GET_QUEUE_TIME = 5
    GET_JOINT_NAMES = 6
    GET_POSITION = 7
    GET_VELOCITY = 8
    GET_TORQUE = 9
    SET_POSITION = 10
    SET_VELOCITY = 11
    SET_TORQUE = 12
    SET_IMPEDANCE_CONTROLLER_PARAMS = 13

class ResponseType(Enum):
    OK = 0
    INVALID_REQUEST = 1

def to_bytes(t, d):
    try:
        if d is None:
            return b''

        d = t(d)
        if isinstance(d, RequestType):
            return struct.pack("!i", d.value)
        if isinstance(d, ResponseType):
            return struct.pack("!i", d.value)
        if isinstance(d, str):
            return str.encode(d)
        if isinstance(d, int):
            return struct.pack("!i", d)
        if isinstance(d, float):
            return struct.pack("!f", d)
    except Exception:
        raise ValueError(f"Could not convert {d} to bytes")
    raise NotImplementedError

def from_bytes(t, d):
    try:
        if len(d) == 0:
            return None
        if t == RequestType:
            return RequestType(struct.unpack("!i", d)[0])
        if t == ResponseType:
            return ResponseType(struct.unpack("!i", d)[0])
        if t == str:
            return d.decode()
        if t == int:
            return struct.unpack("!i", d)[0]
        if t == float:
            return struct.unpack("!f", d)[0]
    except Exception:
        raise ValueError(f"Could not convert {d} to type {t}")
    raise NotImplementedError

class Client:

    def __init__(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("tcp://localhost:4242")

    def start_experiment(
        self,
        user_token: str,
        experiment_type: str,
        requested_time: float
    ) -> tuple[str, str]:
        self._send_request(
            RequestType.START_EXPERIMENT,
            [str, str, float],
            [user_token, experiment_type, requested_time]
        )
        return tuple(self._receive_response([str, str]))

    def stop_experiment(self, session_token: str) -> int:
        self._send_request(
            RequestType.STOP_EXPERIMENT,
            [str],
            [session_token]
        )
        return self._receive_response([int])[0]

    def start_recording(self, session_token: str) -> str:
        self._send_request(
            RequestType.START_RECORDING,
            [str],
            [session_token]
        )
        return self._receive_response([str])[0]

    def stop_recording(self, session_token: str, filename: str) -> str:
        self._send_request(
            RequestType.STOP_RECORDING,
            [str, str],
            [session_token, filename]
        )
        return self._receive_response([str])[0]

    def get_queue_time(self, experiment_type: str) -> float:
        self._send_request(
            RequestType.GET_QUEUE_TIME,
            [str],
            [experiment_type]
        )
        return self._receive_response([float])[0]

    def get_joint_names(self, session_token: str) -> list[str]:
        self._send_request(
            RequestType.GET_JOINT_NAMES,
            [str],
            [session_token]
        )
        return self._receive_response([str])

    def get_position(self, session_token: str) -> list[float]:
        self._send_request(
            RequestType.GET_POSITION,
            [str],
            [session_token]
        )
        res =  self._receive_response([float])
        if len(res) == 1:
            return res[0]
        return res

    def get_velocity(self, session_token: str) -> list[float]:
        self._send_request(
            RequestType.GET_VELOCITY,
            [str],
            [session_token]
        )
        res =  self._receive_response([float])
        if len(res) == 1:
            return res[0]
        return res

    def get_torque(self, session_token: str) -> list[float]:
        self._send_request(
            RequestType.GET_TORQUE,
            [str],
            [session_token]
        )
        res =  self._receive_response([float])
        if len(res) == 1:
            return res[0]
        return res

    def set_position(self, positions: list[float | None] | float, session_token: str):
        if not isinstance(positions, list):
            positions = float(positions)
        if isinstance(positions, float):
            positions = [positions]
        for pos in positions:
            pos = float(pos)
        self._send_request(
            RequestType.SET_POSITION,
            [str] + [float] * len(positions),
            [session_token] + positions
        )
        self._receive_response([])

    def set_velocity(self, velocities: list[float | None] | float, session_token: str):
        if not isinstance(velocities, list):
            velocities = float(velocities)
        if isinstance(velocities, float):
            velocities = [velocities]
        for vel in velocities:
            vel = float(vel)
        self._send_request(
            RequestType.SET_VELOCITY,
            [str] + [float] * len(velocities),
            [str(session_token)] + velocities
        )
        self._receive_response([])

    def set_torque(self, torques: list[float | None] | float, session_token: str):
        if not isinstance(torques, list):
            torques = float(torques)
        if isinstance(torques, float):
            torques  = [torques ]
        for tor in torques :
            tor = float(tor)
        self._send_request(
            RequestType.SET_TORQUE,
            [str] + [float] * len(torques),
            [str(session_token)] + torques
        )
        self._receive_response([])

    def set_impedance_controller_params(
        self,
        kp: float | list[float],
        kd: float | list[float],
        session_token: str
    ):
        if isinstance(kp, float):
            kp = [kp]
        if isinstance(kd, float):
            kd = [kd]

        self._send_request(
            RequestType.SET_IMPEDANCE_CONTROLLER_PARAMS,
            [int, int] + ([float] * len(kp)) + ([float] * len(kd)) + [str],
            [len(kp), len(kd)] + kp + kd + [str(session_token)]
        )
        self._receive_response([])

    def _send_request(self, request_type, data_types, data):
        assert len(data_types) == len(data)
        data_bytes = list(map(lambda d: to_bytes(d[0], d[1]), zip(data_types, data)))
        self.socket.send_multipart([to_bytes(RequestType, request_type)] + data_bytes)

    def _receive_response(self, data_types):
        res = self.socket.recv_multipart()
        response_type = from_bytes(ResponseType, res[0])
        if response_type != ResponseType.OK:
            response_message = from_bytes(str, res[1])
            raise RuntimeError(f"Invalid request: {response_message}")

        res = res[1:]
        if len(data_types) < len(res):
            data_types = data_types + [data_types[-1]] * (len(res) - len(data_types))

        return list(map(lambda d: from_bytes(d[0], d[1]), zip(data_types, res)))
