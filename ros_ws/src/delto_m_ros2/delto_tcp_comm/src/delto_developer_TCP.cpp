// Copyright 2025 TESOLLO
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above copyright
//      notice, this list of conditions and the following disclaimer in the
//      documentation and/or other materials provided with the distribution.
//
//    * Neither the name of the TESOLLO nor the names of its
//      contributors may be used to endorse or promote products derived from
//      this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "delto_tcp_comm/delto_developer_TCP.hpp"

#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <poll.h>
#include <fcntl.h>
#include <cerrno>
#include <cstring>
#include <chrono>
#include <stdexcept>

namespace DeltoTCP {

// ============================================================================
// Helper Functions
// ============================================================================

int Communication::GetMotorCount(uint16_t model) {
  switch (model) {
    case static_cast<uint16_t>(ModelType::DG1F):
      return 3;
    case static_cast<uint16_t>(ModelType::DG2F):
      return 6;
    case static_cast<uint16_t>(ModelType::DG3F_B):
    case static_cast<uint16_t>(ModelType::DG3F_M):
      return 12;
    case static_cast<uint16_t>(ModelType::DG4F):
      return 18;
    case static_cast<uint16_t>(ModelType::DG5F):
    case static_cast<uint16_t>(ModelType::DG5F_L):
    case static_cast<uint16_t>(ModelType::DG5F_R):
    case static_cast<uint16_t>(ModelType::DG5F_L_S):
    case static_cast<uint16_t>(ModelType::DG5F_R_S):
      return 20;
    case static_cast<uint16_t>(ModelType::DG5F_L_S15):
    case static_cast<uint16_t>(ModelType::DG5F_R_S15):
      return 15;
    default:
      std::cerr << "Unknown model type: 0x" << std::hex << model << std::dec
                << std::endl;
      return 12;
  }
}

int Communication::GetBytePerMotor(uint16_t model) {
  if (model == static_cast<uint16_t>(ModelType::DG3F_B)) {
    return 5;
  }
  return 8;
}

int Communication::GetFingerCount(uint16_t model) {
  switch (model) {
    case static_cast<uint16_t>(ModelType::DG1F):
      return 1;
    case static_cast<uint16_t>(ModelType::DG2F):
      return 2;
    case static_cast<uint16_t>(ModelType::DG3F_B):
    case static_cast<uint16_t>(ModelType::DG3F_M):
      return 3;
    case static_cast<uint16_t>(ModelType::DG4F):
      return 4;
    case static_cast<uint16_t>(ModelType::DG5F):
    case static_cast<uint16_t>(ModelType::DG5F_L):
    case static_cast<uint16_t>(ModelType::DG5F_R):
    case static_cast<uint16_t>(ModelType::DG5F_L_S):
    case static_cast<uint16_t>(ModelType::DG5F_R_S):
    case static_cast<uint16_t>(ModelType::DG5F_L_S15):
    case static_cast<uint16_t>(ModelType::DG5F_R_S15):
      return 5;
    default:
      return 5;
  }
}

bool Communication::IsNewModel() const {
  return model_ != static_cast<uint16_t>(ModelType::DG3F_B);
}

bool Communication::SupportsExtendedFeatures() const {
  return model_ != static_cast<uint16_t>(ModelType::DG3F_B);
}

int Communication::GetSensorBytesPerFinger() const {
  switch (sensor_type_) {
    case SensorType::FT_6AXIS:  return 12;
    case SensorType::FT_3AXIS:  return 12;
    case SensorType::FT_4AXIS:  return 12;
    case SensorType::TACTILE_M: return 15;
    case SensorType::TACTILE_S: return 36;
    default:                    return 0;
  }
}

int Communication::GetSensorFingerCount() const {
  return finger_count_;
}

int Communication::CalculateExpectedResponseLength() {
  int length = static_cast<int>(HEADER_SIZE) + motor_count_ * byte_per_motor_;

  if (SupportsExtendedFeatures()) {
    if (fingertip_sensor_ && sensor_type_ != SensorType::NONE) {
      int sensor_fingers = GetSensorFingerCount();
      length += GetSensorBytesPerFinger() * sensor_fingers;
    }
    if (io_) {
      length += GPIO_SIZE;
    }
  }

  return length;
}

int16_t Communication::CombineMsg(uint8_t data1, uint8_t data2) {
  return static_cast<int16_t>(static_cast<uint16_t>(data1 << 8) | data2);
}

int8_t Communication::ConvertByte(uint8_t byte) {
  if (byte >= 0x80) {
    return static_cast<int8_t>(byte - 0x100);
  }
  return static_cast<int8_t>(byte);
}

// ============================================================================
// Low-level I/O
// ============================================================================

bool Communication::SendAll(const uint8_t* data, std::size_t len,
                            int timeout_ms) {
  if (sockfd_ < 0) {
    return false;
  }

  std::size_t sent = 0;
  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::milliseconds(timeout_ms);

  while (sent < len) {
    // Bounded wait for writability so a full send buffer cannot block forever.
    int remaining_ms = static_cast<int>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now()).count());
    if (remaining_ms <= 0) {
      std::cerr << "Send timeout: sent " << sent << "/" << len << " bytes"
                << std::endl;
      return false;
    }

    struct pollfd pfd;
    pfd.fd = sockfd_;
    pfd.events = POLLOUT;

    int ret = ::poll(&pfd, 1, remaining_ms);
    if (ret < 0) {
      if (errno == EINTR) continue;
      std::cerr << "Send poll error: " << std::strerror(errno) << std::endl;
      return false;
    }
    if (ret == 0) {
      std::cerr << "Send timeout: sent " << sent << "/" << len << " bytes"
                << std::endl;
      return false;
    }
    if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) {
      std::cerr << "Socket error during send poll" << std::endl;
      return false;
    }

    ssize_t n = ::send(sockfd_, data + sent, len - sent, MSG_NOSIGNAL);
    if (n < 0) {
      if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK) continue;
      std::cerr << "Send error: " << std::strerror(errno) << std::endl;
      return false;
    }
    if (n == 0) {
      std::cerr << "Send error: connection closed" << std::endl;
      return false;
    }
    sent += static_cast<std::size_t>(n);
  }
  return true;
}

void Communication::DrainSocket() {
  if (sockfd_ < 0) {
    return;
  }

  uint8_t scratch[256];
  std::size_t dropped = 0;
  while (true) {
    ssize_t n = ::recv(sockfd_, scratch, sizeof(scratch), MSG_DONTWAIT);
    if (n <= 0) {
      break;  // EAGAIN (empty), EOF, or error -- nothing more to discard
    }
    dropped += static_cast<std::size_t>(n);
    if (dropped > 64 * 1024) {
      break;  // pathological flood: stop draining rather than spin
    }
  }

  if (dropped > 0) {
    std::cerr << "Dropped " << dropped
              << " stale bytes before request (stream was desynced)"
              << std::endl;
  }
}

bool Communication::RecvAll(uint8_t* data, std::size_t len, int timeout_ms) {
  std::size_t received = 0;
  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::milliseconds(timeout_ms);

  while (received < len) {
    int remaining_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        deadline - std::chrono::steady_clock::now()).count();
    if (remaining_ms <= 0) {
      std::cerr << "Read timeout: got " << received << "/" << len
                << " bytes" << std::endl;
      return false;
    }

    struct pollfd pfd;
    pfd.fd = sockfd_;
    pfd.events = POLLIN;

    int ret = ::poll(&pfd, 1, remaining_ms);
    if (ret < 0) {
      std::cerr << "Poll error: " << std::strerror(errno) << std::endl;
      return false;
    }
    if (ret == 0) {
      std::cerr << "Read timeout: got " << received << "/" << len
                << " bytes" << std::endl;
      return false;
    }
    if (pfd.revents & (POLLERR | POLLHUP | POLLNVAL)) {
      std::cerr << "Socket error during poll" << std::endl;
      return false;
    }

    ssize_t n = ::recv(sockfd_, data + received, len - received, 0);
    if (n <= 0) {
      std::cerr << "Read error: " << (n == 0 ? "Connection closed" :
                    std::strerror(errno)) << std::endl;
      return false;
    }
    received += static_cast<std::size_t>(n);
  }
  return true;
}

// ============================================================================
// Constructor / Destructor
// ============================================================================

Communication::Communication(const std::string& ip, int port, uint16_t model,
                             bool fingertip_sensor, bool io)
    : ip_(ip),
      port_(port),
      model_(model),
      actual_model_(0),
      fingertip_sensor_(fingertip_sensor),
      io_(io),
      sensor_type_(SensorType::NONE),
      finger_sensor_mask_(0),
      sockfd_(-1),
      motor_count_(GetMotorCount(model)),
      byte_per_motor_(GetBytePerMotor(model)),
      finger_count_(GetFingerCount(model)),
      expected_response_length_(0) {}

Communication::~Communication() {
  Disconnect();
}

std::string Communication::ModelToString(uint16_t model) {
  switch (model) {
    case static_cast<uint16_t>(ModelType::DG1F):
      return "DG1F (0x1F02)";
    case static_cast<uint16_t>(ModelType::DG2F):
      return "DG2F-M (0x2F02)";
    case static_cast<uint16_t>(ModelType::DG3F_B):
      return "DG3F-B (0x3F01)";
    case static_cast<uint16_t>(ModelType::DG3F_M):
      return "DG3F-M (0x3F02)";
    case static_cast<uint16_t>(ModelType::DG4F):
      return "DG4F (0x4F02)";
    case static_cast<uint16_t>(ModelType::DG5F):
      return "DG5F (0x5F02)";
    case static_cast<uint16_t>(ModelType::DG5F_L):
      return "DG5F-L (0x5F12)";
    case static_cast<uint16_t>(ModelType::DG5F_R):
      return "DG5F-R (0x5F22)";
    case static_cast<uint16_t>(ModelType::DG5F_L_S):
      return "DG5F-L-S (0x5F14)";
    case static_cast<uint16_t>(ModelType::DG5F_R_S):
      return "DG5F-R-S (0x5F24)";
    case static_cast<uint16_t>(ModelType::DG5F_L_S15):
      return "DG5F-L-S15 (0x5F34)";
    case static_cast<uint16_t>(ModelType::DG5F_R_S15):
      return "DG5F-R-S15 (0x5F44)";
    default:
      return "Unknown (0x" +
             ([](uint16_t v) {
               char buf[8];
               snprintf(buf, sizeof(buf), "%04X", v);
               return std::string(buf);
             })(model) + ")";
  }
}

// ============================================================================
// Connection Management
// ============================================================================

void Communication::Connect() {
  // Close existing socket
  Disconnect();

  sockfd_ = ::socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd_ < 0) {
    throw std::runtime_error("Failed to create socket: " +
                             std::string(std::strerror(errno)));
  }

  struct sockaddr_in addr;
  std::memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(static_cast<uint16_t>(port_));
  if (::inet_pton(AF_INET, ip_.c_str(), &addr.sin_addr) <= 0) {
    ::close(sockfd_);
    sockfd_ = -1;
    throw std::runtime_error("Invalid IP address: " + ip_);
  }

  // Non-blocking connect with an explicit timeout: a blocking ::connect() to an
  // unreachable host sits in the kernel SYN retry for ~2 minutes.
  {
    constexpr int CONNECT_TIMEOUT_MS = 3000;

    int flags = ::fcntl(sockfd_, F_GETFL, 0);
    if (flags < 0 || ::fcntl(sockfd_, F_SETFL, flags | O_NONBLOCK) < 0) {
      ::close(sockfd_);
      sockfd_ = -1;
      throw std::runtime_error("Failed to set socket non-blocking: " +
                               std::string(std::strerror(errno)));
    }

    int rc = ::connect(sockfd_, reinterpret_cast<struct sockaddr*>(&addr),
                       sizeof(addr));
    if (rc < 0 && errno != EINPROGRESS) {
      int saved = errno;
      ::close(sockfd_);
      sockfd_ = -1;
      throw std::runtime_error("Connection failed to " + ip_ + ":" +
                               std::to_string(port_) + ": " +
                               std::strerror(saved));
    }

    if (rc < 0) {
      struct pollfd pfd;
      pfd.fd = sockfd_;
      pfd.events = POLLOUT;

      int pret = ::poll(&pfd, 1, CONNECT_TIMEOUT_MS);
      if (pret <= 0) {
        ::close(sockfd_);
        sockfd_ = -1;
        throw std::runtime_error(
            "Connection to " + ip_ + ":" + std::to_string(port_) +
            (pret == 0 ? " timed out after " +
                             std::to_string(CONNECT_TIMEOUT_MS) + " ms"
                       : std::string(" failed in poll: ") +
                             std::strerror(errno)));
      }

      int soerr = 0;
      socklen_t soerr_len = sizeof(soerr);
      if (::getsockopt(sockfd_, SOL_SOCKET, SO_ERROR, &soerr, &soerr_len) < 0 ||
          soerr != 0) {
        int saved = (soerr != 0) ? soerr : errno;
        ::close(sockfd_);
        sockfd_ = -1;
        throw std::runtime_error("Connection failed to " + ip_ + ":" +
                                 std::to_string(port_) + ": " +
                                 std::strerror(saved));
      }
    }

    // Back to blocking mode; SendAll/RecvAll poll for readiness themselves.
    if (::fcntl(sockfd_, F_SETFL, flags) < 0) {
      ::close(sockfd_);
      sockfd_ = -1;
      throw std::runtime_error("Failed to restore socket flags: " +
                               std::string(std::strerror(errno)));
    }
  }

  // TCP keepalive for cable-disconnect detection
  int keepalive = 1;
  setsockopt(sockfd_, SOL_SOCKET, SO_KEEPALIVE, &keepalive, sizeof(keepalive));
  int keepidle = 1;
  int keepintvl = 1;
  int keepcnt = 3;
  setsockopt(sockfd_, IPPROTO_TCP, TCP_KEEPIDLE, &keepidle, sizeof(keepidle));
  setsockopt(sockfd_, IPPROTO_TCP, TCP_KEEPINTVL, &keepintvl, sizeof(keepintvl));
  setsockopt(sockfd_, IPPROTO_TCP, TCP_KEEPCNT, &keepcnt, sizeof(keepcnt));

  // Disable Nagle's algorithm for low-latency
  int nodelay = 1;
  setsockopt(sockfd_, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));

  // Get firmware version and model info
  {
    uint8_t request[3] = {0x00, 0x03, GET_VERSION_CMD};
    if (!SendAll(request, 3)) {
      Disconnect();
      throw std::runtime_error("Failed to send version request");
    }

    // Read length field (2 bytes)
    uint8_t len_buf[2];
    if (!RecvAll(len_buf, 2, 3000)) {
      Disconnect();
      throw std::runtime_error("Failed to read device info");
    }

    uint16_t resp_length = (static_cast<uint16_t>(len_buf[0]) << 8) | len_buf[1];

    // Validate before subtracting: length 0 would underflow to 65534, and
    // any length below 7 leaves the model/firmware reads below out of bounds.
    constexpr uint16_t MIN_VERSION_RESP_LENGTH = 7;
    constexpr uint16_t MAX_VERSION_RESP_LENGTH = 64;
    if (resp_length < MIN_VERSION_RESP_LENGTH ||
        resp_length > MAX_VERSION_RESP_LENGTH) {
      Disconnect();
      throw std::runtime_error(
          "Invalid device info length: " + std::to_string(resp_length) +
          " (expected " + std::to_string(MIN_VERSION_RESP_LENGTH) + ".." +
          std::to_string(MAX_VERSION_RESP_LENGTH) + ")");
    }

    uint16_t remaining = static_cast<uint16_t>(resp_length - 2);

    std::vector<uint8_t> payload(remaining);
    if (!RecvAll(payload.data(), remaining, 3000)) {
      Disconnect();
      throw std::runtime_error("Failed to read device info payload");
    }

    actual_model_ = CombineMsg(payload[1], payload[2]);
    firmware_version_ = {payload[3], payload[4]};

    if (resp_length >= 9) {
      sensor_type_ = static_cast<SensorType>(payload[5]);
      finger_sensor_mask_ = payload[6];
    } else {
      sensor_type_ = fingertip_sensor_ ? SensorType::FT_6AXIS : SensorType::NONE;
      finger_sensor_mask_ = fingertip_sensor_ ?
          static_cast<uint8_t>((1 << finger_count_) - 1) : 0x00;
    }

    std::cout << "========================================" << std::endl;
    std::cout << "Device Info:" << std::endl;
    std::cout << "  Firmware Version: "
              << static_cast<int>(firmware_version_[0]) << "."
              << static_cast<int>(firmware_version_[1]) << std::endl;
    std::cout << "  Configured Model: " << ModelToString(model_) << std::endl;
    std::cout << "  Actual Model:     " << ModelToString(actual_model_) << std::endl;

    if (resp_length >= 9) {
      std::cout << "  Sensor Type:      0x" << std::hex
                << static_cast<int>(payload[5]) << std::dec << std::endl;
      std::string mask_bin = "0b";
      for (int b = 7; b >= 0; --b) {
        mask_bin += (finger_sensor_mask_ & (1 << b)) ? '1' : '0';
      }
      std::cout << "  Finger Mask:      " << mask_bin << std::endl;
      std::cout << "  Finger Sensor:    ";
      for (int i = finger_count_ - 1; i >= 0; --i) {
        bool equipped = finger_sensor_mask_ & (1 << i);
        std::cout << "F" << (i + 1) << ":" << (equipped ? "ON" : "--");
        if (i > 0) std::cout << " ";
      }
      std::cout << std::endl;
    }

    if (model_ != actual_model_) {
      std::cerr << "========================================" << std::endl;
      std::cerr << "WARNING: Model mismatch detected!" << std::endl;
      std::cerr << "  Expected: " << ModelToString(model_) << std::endl;
      std::cerr << "  Got:      " << ModelToString(actual_model_) << std::endl;
      std::cerr << "  Please check your URDF/xacro configuration." << std::endl;
      std::cerr << "========================================" << std::endl;

      // A mismatch that changes the actuator count mis-frames every response,
      // so fail at connect time instead of flapping at runtime.
      const int actual_motor_count = GetMotorCount(actual_model_);
      const int actual_byte_per_motor = GetBytePerMotor(actual_model_);
      if (actual_motor_count != motor_count_ ||
          actual_byte_per_motor != byte_per_motor_) {
        Disconnect();
        throw std::runtime_error(
            "Model mismatch changes the frame layout: configured " +
            ModelToString(model_) + " (" + std::to_string(motor_count_) +
            " motors x " + std::to_string(byte_per_motor_) +
            " B) vs device " + ModelToString(actual_model_) + " (" +
            std::to_string(actual_motor_count) + " motors x " +
            std::to_string(actual_byte_per_motor) +
            " B). Fix delto_model in the URDF/xacro.");
      }
    } else {
      std::cout << "  Status:           OK (Model matched)" << std::endl;
    }
    std::cout << "========================================" << std::endl;
  }

  expected_response_length_ = CalculateExpectedResponseLength();

  std::cout << "Connected to Delto Gripper (Model: " << ModelToString(actual_model_)
            << ", Motors: " << motor_count_ << ")" << std::endl;
}

void Communication::Disconnect() {
  if (sockfd_ >= 0) {
    ::close(sockfd_);
    sockfd_ = -1;
    std::cout << "Disconnected from Delto Gripper" << std::endl;
  }
}

// ============================================================================
// Data Exchange
// ============================================================================

bool Communication::ReadFullPacket(std::vector<uint8_t>& buffer) {
  // RecvAll(ptr, 0) trivially succeeds, which would leave the header reads in
  // GetData() out of bounds on an empty buffer.
  if (expected_response_length_ <= static_cast<int>(HEADER_SIZE)) {
    std::cerr << "Invalid expected response length: "
              << expected_response_length_ << std::endl;
    return false;
  }
  // Frame by the device-declared length: firmware >= 3.0 appends an error
  // code section after the requested data, so the response can be longer
  // than the minimum computed from the model/sensor configuration.
  uint8_t len_buf[2];
  if (!RecvAll(len_buf, 2, 500)) {
    return false;
  }

  const int declared_length = (static_cast<int>(len_buf[0]) << 8) | len_buf[1];
  if (declared_length < expected_response_length_ ||
      declared_length > static_cast<int>(MAX_RESPONSE_LENGTH)) {
    std::cerr << "Invalid response length: device declared " << declared_length
              << " but driver expects at least " << expected_response_length_
              << " (model/sensor configuration mismatch)" << std::endl;
    return false;
  }

  buffer.resize(static_cast<std::size_t>(declared_length));
  buffer[0] = len_buf[0];
  buffer[1] = len_buf[1];
  return RecvAll(buffer.data() + 2,
                 static_cast<std::size_t>(declared_length) - 2, 500);
}

DeltoReceivedData Communication::GetData() {
 
  // Build request packet
  // JOINT            = 0x01,
  // CURRENT          = 0x02,
  // TEMPERATURE      = 0x03,
  // VELOCITY         = 0x04,
  // FINGERTIP_SENSOR = 0x05,
  // GPIO             = 0x06,
  // MODULE_ERROR     = 0x07,
  std::vector<uint8_t> request;

  if (model_ == static_cast<uint16_t>(ModelType::DG3F_B)) {
    request = {0x00, 0x05, GET_DATA_CMD, static_cast<uint8_t>(DataCode::JOINT), static_cast<uint8_t>(DataCode::CURRENT)};
  } else {
    request = {0x00, 0x07, GET_DATA_CMD, static_cast<uint8_t>(DataCode::JOINT), static_cast<uint8_t>(DataCode::CURRENT), static_cast<uint8_t>(DataCode::TEMPERATURE), static_cast<uint8_t>(DataCode::VELOCITY)};

    if (SupportsExtendedFeatures()) {
      if (fingertip_sensor_) {
        request.push_back(static_cast<uint8_t>(DataCode::FINGERTIP_SENSOR));
      }
      if (io_) {
        request.push_back(static_cast<uint8_t>(DataCode::GPIO));
      }
      // Firmware >= 3.0 reports a trailing error code
      if (firmware_version_.size() >= 2 && firmware_version_[0] >= 3) {
        request.push_back(static_cast<uint8_t>(DataCode::MODULE_ERROR));
      }
      request[1] = static_cast<uint8_t>(request.size());
    }
  }

  // No transaction id in this protocol: anything already queued is a stale
  // reply, and reading it would shift every field by a whole frame.
  DrainSocket();

  // Send request
  if (!SendAll(request.data(), request.size())) {
    throw std::runtime_error("Failed to send GetData request");
  }

  // Read response
  std::vector<uint8_t> response;
  if (!ReadFullPacket(response)) {
    throw std::runtime_error("Failed to read full packet from gripper");
  }

  // ReadFullPacket framed the buffer by the device-declared length and
  // already rejected anything below the configured minimum.
  const uint8_t cmd = response[2];

  if (cmd != GET_DATA_CMD) {
    throw std::runtime_error(
        "Invalid response: CMD 0x" + std::to_string(static_cast<int>(cmd)) +
        " (expected 0x01)");
  }

  // Parse motor data
  DeltoReceivedData received_data;
  received_data.joint.resize(motor_count_);
  received_data.current.resize(motor_count_);

  if (IsNewModel()) {
    received_data.velocity.resize(motor_count_);
    received_data.temperature.resize(motor_count_);
  }

  for (int i = 0; i < motor_count_; i++) {
    size_t base = HEADER_SIZE + i * byte_per_motor_;
    [[maybe_unused]] uint8_t motor_id = response[base];

    // Big-endian on the wire: first byte of each field is the HIGH byte.
    uint8_t pos_hi = response[base + 1];
    uint8_t pos_lo = response[base + 2];
    uint8_t cur_hi = response[base + 3];
    uint8_t cur_lo = response[base + 4];

    int16_t raw_position = CombineMsg(pos_hi, pos_lo);
    int16_t raw_current = CombineMsg(cur_hi, cur_lo);

    if(actual_model_ == static_cast<uint16_t>(ModelType::DG5F_L_S) 
    || actual_model_ == static_cast<uint16_t>(ModelType::DG5F_R_S)
    || actual_model_ == static_cast<uint16_t>(ModelType::DG5F_L_S15)
    || actual_model_ == static_cast<uint16_t>(ModelType::DG5F_R_S15))
  {
    received_data.joint[i] = raw_position * POSITION_SCALE_S;
  } else {
    received_data.joint[i] = raw_position * POSITION_SCALE;
  }
  
    received_data.current[i] = raw_current * CURRENT_SCALE;

    if (IsNewModel()) {
      uint8_t temp_hi = response[base + 5];
      uint8_t temp_lo = response[base + 6];
      int16_t raw_temperature = CombineMsg(temp_hi, temp_lo);
      received_data.temperature[i] = raw_temperature * 0.1;

      int8_t raw_vel = ConvertByte(response[base + 7]);
      received_data.velocity[i] =
          static_cast<double>(raw_vel) * VELOCITY_SCALE;
    }
  }

  // Parse sensor data
  if (SupportsExtendedFeatures() && fingertip_sensor_ && sensor_type_ != SensorType::NONE) {
    size_t sensor_base = HEADER_SIZE + motor_count_ * byte_per_motor_;
    int bytes_per_finger = GetSensorBytesPerFinger();

    switch (sensor_type_) {
      case SensorType::FT_6AXIS:
      case SensorType::FT_3AXIS:
      case SensorType::FT_4AXIS: {
        received_data.fingertip_sensor.resize(finger_count_ * 6);
        for (int finger = 0; finger < finger_count_; finger++) {
          for (int axis = 0; axis < 6; axis++) {
            size_t offset = sensor_base + finger * bytes_per_finger + axis * 2;
            int16_t raw_value = CombineMsg(response[offset], response[offset + 1]);
            
            // 0.1N to N for force, 1 mNm to Nm for torque
            if (axis < 3) {
              received_data.fingertip_sensor[finger * 6 + axis] = raw_value * 0.1;
            } else {
              received_data.fingertip_sensor[finger * 6 + axis] = raw_value * 0.001;
            }
          }
        }
        break;
      }
      case SensorType::TACTILE_M: {
        for (int finger = 0; finger < finger_count_; finger++) {
          std::vector<uint8_t> cells(15);
          size_t offset = sensor_base + finger * bytes_per_finger;
          for (int j = 0; j < 15; j++) {
            cells[j] = response[offset + j];
          }
          received_data.tactile_m.push_back(std::move(cells));
        }
        break;
      }
      case SensorType::TACTILE_S: {
        for (int finger = 0; finger < finger_count_; finger++) {
          std::vector<uint16_t> cells(18);
          size_t offset = sensor_base + finger * bytes_per_finger;
          for (int j = 0; j < 18; j++) {
            cells[j] = (static_cast<uint16_t>(response[offset + j * 2]) << 8) |
                        response[offset + j * 2 + 1];
          }
          received_data.tactile_s.push_back(std::move(cells));
        }
        break;
      }
      default:
        break;
    }
  }

  // Parse GPIO data
  if (SupportsExtendedFeatures() && io_) {
    size_t gpio_base = HEADER_SIZE + motor_count_ * byte_per_motor_;
    if (fingertip_sensor_ && sensor_type_ != SensorType::NONE) {
      gpio_base += GetSensorBytesPerFinger() * GetSensorFingerCount();
    }

    received_data.gpio.resize(4);
    for (int i = 0; i < 4; i++) {
      received_data.gpio[i] = (response[gpio_base + i] != 0);
    }
  }

  // Parse error code (firmware >= 3.0, requested via MODULE_ERROR): a 2-byte
  // big-endian decimal appended after all requested sections.
  // 400 + error number; 0 = OK.
  if (SupportsExtendedFeatures() && firmware_version_.size() >= 2 &&
      firmware_version_[0] >= 3) {
    size_t error_base = HEADER_SIZE + motor_count_ * byte_per_motor_;
    if (fingertip_sensor_ && sensor_type_ != SensorType::NONE) {
      error_base += GetSensorBytesPerFinger() * GetSensorFingerCount();
    }
    if (io_) {
      error_base += GPIO_SIZE;
    }

    if (response.size() >= error_base + 2) {
      const uint16_t error_code =
          (static_cast<uint16_t>(response[error_base]) << 8) |
          response[error_base + 1];
      if (error_code != 0) {
        std::cerr << "[Error] error code : " << error_code << std::endl;
      }
    }
  }

  return received_data;
}

void Communication::SendDuty(std::vector<int>& duty) {
  // Never read past the end of the caller's vector: that would send
  // uninitialised heap to the motors as a duty command.
  if (duty.size() < static_cast<std::size_t>(motor_count_)) {
    throw std::runtime_error(
        "SendDuty: duty vector has " + std::to_string(duty.size()) +
        " entries but this model has " + std::to_string(motor_count_) +
        " motors");
  }

  uint16_t total_packet_size = 3 + motor_count_ * 3;
  std::vector<uint8_t> tcp_data_send(total_packet_size);

  tcp_data_send[0] = (total_packet_size >> 8) & 0xFF;
  tcp_data_send[1] = (total_packet_size) & 0xFF;
  tcp_data_send[2] = SET_DUTY_CMD;

  for (int i = 0; i < motor_count_; ++i) {
    tcp_data_send[3 + i * 3] = i + 1;
    tcp_data_send[4 + i * 3] = (duty[i] >> 8) & 0xFF;
    tcp_data_send[5 + i * 3] = (duty[i]) & 0xFF;
  }

  if (!SendAll(tcp_data_send.data(), tcp_data_send.size())) {
    throw std::runtime_error("Failed to send duty command");
  }
}

// ============================================================================
// GPIO Control
// ============================================================================

void Communication::SetGPIO(bool output1, bool output2, bool output3) {
  if (!SupportsExtendedFeatures()) {
    std::cerr << "GPIO not supported on this model" << std::endl;
    return;
  }

  uint8_t request[6] = {
    0x00, 0x06, SET_GPIO_CMD,
    static_cast<uint8_t>(output1 ? 0x01 : 0x00),
    static_cast<uint8_t>(output2 ? 0x01 : 0x00),
    static_cast<uint8_t>(output3 ? 0x01 : 0x00)
  };

  if (!SendAll(request, 6)) {
    std::cerr << "Error sending GPIO command" << std::endl;
  }
}

// ============================================================================
// F/T Sensor Control
// ============================================================================

void Communication::SetFTSensorOffset() {
  if (!SupportsExtendedFeatures()) {
    std::cerr << "F/T sensor not supported on this model" << std::endl;
    return;
  }

  uint8_t request[3] = {0x00, 0x03, SET_FT_SENSOR_OFFSET_CMD};

  if (!SendAll(request, 3)) {
    std::cerr << "Error sending F/T sensor offset command" << std::endl;
  }
}

// ============================================================================
// Version Info
// ============================================================================

std::vector<uint8_t> Communication::GetFirmwareVersion() {
  return firmware_version_;
}

}  // namespace DeltoTCP
