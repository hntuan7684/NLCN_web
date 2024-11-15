// Fetch and update video path based on the selected video
function fetchVideo() {
  const videoChoice = document.getElementById("video-choice").value;
  fetch(`?vi_id=${videoChoice}`)
    .then((response) => response.json())
    .then((data) => {
      if (data.status === "success") {
        const videoElement = document.getElementById("traffic-video");
        const videoSource = document.getElementById("video-source");
        videoSource.src = data.video_path;
        videoElement.load();
        videoElement.play();
      } else {
        alert("Error loading video");
      }
    })
    .catch((error) => {
      console.error("Error:", error);
      alert("Failed to fetch video. Please try again.");
    });
}

// Lắng nghe sự kiện khi người dùng nhấn vào một vi phạm trong danh sách
document.addEventListener("DOMContentLoaded", function () {
  const buttons = document.querySelectorAll(".violation-list .btn");

  buttons.forEach((button) => {
    button.addEventListener("click", function () {
      // Lấy dữ liệu từ thuộc tính data-* của button
      const violationId = button.getAttribute("data-vt-id");
      const licenseNumber = button.getAttribute("data-bs-license");
      const violationType = button.getAttribute("data-bs-violations");
      const violationImage = button.getAttribute("data-bs-images");
      const violationSpeed = button.getAttribute("data-bs-speed")
      const violationTime = button.getAttribute("data-bs-time"); // Thêm dữ liệu thời gian vi phạm nếu có

      // Cập nhật nội dung trong modal
      document.getElementById("violation-id").textContent = violationId;
      document.getElementById("license-number").textContent = licenseNumber;
      document.getElementById("violation-type").textContent = violationType;
      document.getElementById("violation-speed").textContent = violationSpeed;
      document.getElementById("violation-time").textContent = violationTime;
      document.getElementById("violation-image").src = violationImage; // Cập nhật hình ảnh vi phạm
    });
  });
});

// Lấy phần tử video
const videoElement = document.getElementById("traffic-video");

// Hàm xử lý Pause
function pauseVideo() {
  videoElement.pause(); // Dừng video
  document.getElementById("pauseButton").disabled = true; // Vô hiệu hóa nút Pause
  document.getElementById("continueButton").disabled = false; // Kích hoạt nút Continue
}

// Hàm xử lý Continue
function continueVideo() {
  videoElement.play(); // Tiếp tục phát video
  document.getElementById("continueButton").disabled = true; // Vô hiệu hóa nút Continue
  document.getElementById("pauseButton").disabled = false; // Kích hoạt nút Pause
}

// Optional: Add event listeners for video status updates
videoElement.addEventListener("ended", function () {
  document.getElementById("continueButton").disabled = true; // Disable continue button if video ends
});
videoElement.addEventListener("play", function () {
  document.getElementById("pauseButton").disabled = false; // Enable pause button when video is playing
});
videoElement.addEventListener("pause", function () {
  document.getElementById("continueButton").disabled = false; // Enable continue button when video is paused
});
