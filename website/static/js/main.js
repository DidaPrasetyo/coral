document.addEventListener("DOMContentLoaded", function () {
    const videoElement = document.getElementById("video-stream");
    const urlForm = document.getElementById("url-form");

    urlForm.addEventListener("submit", function (event) {
        event.preventDefault();
        const newUrl = document.getElementById("url").value;
        
        // Update the video source
        videoElement.src = newUrl;

        // Reset the form
        urlForm.reset();
    });
});
