// Fetch and update video path based on the selected video
function fetchVideo() {
  const videoChoice = document.getElementById("video-choice").value;
  fetch(`/get_video?vi_id=${videoChoice}`) // Update route to match your backend
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
      const violationSpeed = button.getAttribute("data-bs-speed");
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

// WebSocket connection to receive updates
var socket = io.connect("http://" + document.domain + ":" + location.port);

socket.on("update_violations", function (data) {
  var tableBody = document.querySelector("#violations tbody");
  tableBody.innerHTML = ""; // Clear the current table
  data.forEach(function (violation) {
    var row = document.createElement("tr");
    var cell1 = document.createElement("td");
    var cell2 = document.createElement("td");
    var cell3 = document.createElement("td");
    cell1.textContent = violation.license_number; // License Number
    cell2.textContent = violation.speed; // Speed
    cell3.textContent = violation.timestamp; // Violation Time
    row.appendChild(cell1);
    row.appendChild(cell2);
    row.appendChild(cell3);
    tableBody.appendChild(row);
  });
});
