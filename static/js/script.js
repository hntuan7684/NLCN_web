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

function fetchViolations() {
  $.ajax({
    url: "/get_violations",
    method: "GET",
    success: function (data) {
      let tableBody = $("#violations-table tbody");
      tableBody.empty(); // Clear the table before adding new data

      data.forEach(function (violation) {
        tableBody.append(
          `<tr>
            <td>${violation.license_number}</td>
            <td><img src="static/images/${violation.image}" alt="Violation Image" width="100"></td>
            <td>${violation.violation_type}</td>
            <td>${violation.speed} km/h</td>
            <td>${violation.timestamp}</td>
          </tr>`
        );
      });
    },
    error: function (error) {
      console.error("Error fetching violations:", error);
    },
  });
}

// Fetch violations every 5 seconds
setInterval(fetchViolations, 5000);

// Initial fetch on page load
fetchViolations();
