"""
DDS Camera Wrapper for receiving video stream via DDS
"""
import threading
import time
import numpy as np
import av
import cv2
import io
from enum import IntEnum
from dataclasses import dataclass
from typing import Optional, Tuple
from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.sub import Subscriber, DataReader
from cyclonedds.core import Qos, Policy
from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import uint64, sequence, uint8

from orca_gym.log.orca_log import get_orca_logger
_logger = get_orca_logger()


# ==================== DDS 数据类型定义 ====================

class FrameType(IntEnum):
    """视频帧类型枚举"""
    IDR = 0  # 关键帧
    P = 1    # 预测帧
    B = 2    # 双向预测帧


@dataclass
class VideoStream(IdlStruct, typename="VideoStream"):
    """
    视频流数据结构（对应服务端的VideoStream）
    """
    m_index: uint64 = 0
    m_machineID: str = ""
    m_streamID: str = ""
    m_frameType: int = 0  # FrameType
    m_videoStream: sequence[uint8] = None
    
    def __post_init__(self):
        if self.m_videoStream is None:
            self.m_videoStream = []


@dataclass
class RequestIDR(IdlStruct, typename="RequestIDR"):
    """
    IDR关键帧请求数据结构（对应服务端的RequestIDR）
    """
    m_videoStreamID: Optional[str] = None


# ==================== DDS Camera Wrapper ====================

class DDSCameraWrapper:
    """
    DDS视频流接收器
    仿照CameraWrapper的接口和行为，使用DDS接收视频流
    """
    
    def __init__(self, name: str, stream_id: str, topic_name: str = "CameraStreamTopic", domain_id: int = 0):
        """
        初始化DDS相机包装器
        
        Args:
            name: 相机名称（如"camera_head"）
            stream_id: 流ID，用于过滤视频流
            topic_name: DDS主题名称（默认"CameraStreamTopic"）
            domain_id: DDS域ID（默认0）
        """
        self._name = name
        self.stream_id = stream_id
        self.topic_name = topic_name
        self.domain_id = domain_id
        
        # 状态变量（与CameraWrapper保持一致）
        self.image = np.random.randint(0, 255, size=(480, 640, 3), dtype=np.uint8)
        self.enabled = True
        self.received_first_frame = False
        self.image_index = 0
        self.running = False
        
        # DDS相关对象
        self.participant: Optional[DomainParticipant] = None
        self.topic: Optional[Topic] = None
        self.subscriber: Optional[Subscriber] = None
        self.reader: Optional[DataReader] = None
        
        # 视频解码相关
        self.video_buffer = io.BytesIO()
        self.container: Optional[av.container.InputContainer] = None
        self.current_pos = 0
        self.packet_count = 0
        
        # 线程相关
        self.thread: Optional[threading.Thread] = None
        
        print(f"[DDSCameraWrapper][{self.name}] Initialized with stream_id={stream_id}, topic={topic_name}")
    
    @property
    def name(self):
        return self._name
    
    def start(self):
        """启动DDS接收线程"""
        if not self.enabled:
            return
        self.running = True
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()
        print(f"[DDSCameraWrapper][{self.name}] Thread started")
    
    def stop(self):
        """停止DDS接收"""
        if not self.enabled:
            return
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
        self._cleanup_dds()
        print(f"[DDSCameraWrapper][{self.name}] Stopped")
    
    def loop(self):
        """线程入口，运行DDS接收循环"""
        try:
            self.dds_receive_loop()
        except Exception as e:
            print(f"[DDSCameraWrapper][{self.name}] Loop error: {e}")
            import traceback
            traceback.print_exc()
    
    def dds_receive_loop(self):
        """
        DDS主接收循环
        """
        print(f"[DDSCameraWrapper][{self.name}] ========== Initializing DDS ==========")
        
        try:
            # 1. 创建DomainParticipant
            self.participant = DomainParticipant(self.domain_id)
            print(f"[DDSCameraWrapper][{self.name}] ✓ DomainParticipant created (domain_id={self.domain_id})")
            
            # 2. 创建Topic
            self.topic = Topic(self.participant, self.topic_name, VideoStream)
            print(f"[DDSCameraWrapper][{self.name}] ✓ Topic created: {self.topic_name}")
            
            # 3. 配置QoS（RELIABLE可靠性）
            qos = Qos(
                Policy.Reliability.Reliable(max_blocking_time=1000000000),  # 1秒
                Policy.History.KeepLast(10),
                Policy.Durability.Volatile
            )
            
            # 4. 创建Subscriber和DataReader
            self.subscriber = Subscriber(self.participant)
            self.reader = DataReader(self.subscriber, self.topic, qos=qos)
            print(f"[DDSCameraWrapper][{self.name}] ✓ DataReader created with RELIABLE QoS")
            print(f"[DDSCameraWrapper][{self.name}] ========== Waiting for video stream data ==========")
            
            # 5. 轮询接收数据（类似WebSocket的recv循环）
            no_data_count = 0
            last_warning_time = time.time()
            
            while self.running:
                # 读取数据（非阻塞）
                samples = self.reader.take(N=10)  # 每次读取最多10个样本
                
                if samples:
                    no_data_count = 0
                    for sample in samples:
                        self.on_video_stream_received(sample)
                else:
                    # 没有数据，等待一小段时间
                    time.sleep(0.01)  # 10ms
                    no_data_count += 1
                    
                    # 每5秒输出一次警告
                    if no_data_count >= 500 and (time.time() - last_warning_time) >= 5.0:
                        print(f"[DDSCameraWrapper][{self.name}] ⚠️ No data received for 5 seconds...")
                        last_warning_time = time.time()
                        no_data_count = 0
            
        except Exception as e:
            print(f"[DDSCameraWrapper][{self.name}] ERROR in DDS loop: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._cleanup_dds()
    
    def on_video_stream_received(self, video_stream: VideoStream):
        """
        处理接收到的VideoStream数据
        
        Args:
            video_stream: VideoStream对象
        """
        try:
            # 1. 检查stream_id是否匹配
            if video_stream.m_streamID != self.stream_id:
                return  # 忽略不匹配的流
            
            # 2. 获取H.264编码数据
            h264_data = bytes(video_stream.m_videoStream)
            
            # 3. 日志输出
            frame_index = video_stream.m_index
            frame_type = FrameType(video_stream.m_frameType).name if video_stream.m_frameType in [0, 1, 2] else "UNKNOWN"
            
            print(f"[数据长度]: {len(h264_data)} bytes")
            print(f"[DDSCameraWrapper][{self.name}] ✓ Frame #{frame_index}, type={frame_type}, stream_id={video_stream.m_streamID}")
            
            # 4. 写入缓冲区
            self.video_buffer.write(h264_data)
            self.video_buffer.seek(self.current_pos)
            
            # 5. 首次接收时初始化视频容器
            if self.current_pos == 0:
                self.container = av.open(self.video_buffer, mode='r')
                print(f"[DDSCameraWrapper][{self.name}] Video container opened")
            
            # 6. 解码H.264数据
            packets_decoded = 0
            if self.container:
                for packet in self.container.demux():
                    if packet.size == 0:
                        continue
                    
                    frames = packet.decode()
                    for frame in frames:
                        # 更新图像
                        self.image = frame.to_ndarray(format='bgr24')
                        self.image_index += 1
                        packets_decoded += 1
                        
                        # 标记第一帧接收成功
                        if not self.received_first_frame:
                            self.received_first_frame = True
                            print(f"[DDSCameraWrapper][{self.name}] ========== ✓✓✓ FIRST FRAME DECODED! Frame #{self.image_index} ==========")
            
            # 7. 更新缓冲区位置
            self.current_pos += len(h264_data)
            self.packet_count += 1
            
        except Exception as e:
            print(f"[DDSCameraWrapper][{self.name}] Error processing video stream: {e}")
            import traceback
            traceback.print_exc()
    
    def get_frame(self, format='bgr24', size: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, int]:
        """
        获取当前帧（与CameraWrapper接口保持一致）
        
        Args:
            format: 图像格式（'bgr24' 或 'rgb24'）
            size: 目标尺寸 (width, height)，None表示不缩放
            
        Returns:
            (frame, frame_index): 图像帧和帧索引
        """
        if format == 'bgr24':
            frame = self.image
        elif format == 'rgb24':
            frame = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        else:
            frame = self.image
        
        if size is not None:
            frame = cv2.resize(frame, size)
        
        return frame, self.image_index
    
    def is_first_frame_received(self):
        """检查是否已接收第一帧"""
        return self.received_first_frame
    
    def _cleanup_dds(self):
        """清理DDS资源"""
        try:
            if self.reader:
                self.reader = None
            if self.subscriber:
                self.subscriber = None
            if self.topic:
                self.topic = None
            if self.participant:
                self.participant = None
            print(f"[DDSCameraWrapper][{self.name}] DDS resources cleaned up")
        except Exception as e:
            print(f"[DDSCameraWrapper][{self.name}] Error cleaning up DDS: {e}")
    
    def __del__(self):
        """析构函数"""
        if self.running:
            self.stop()
