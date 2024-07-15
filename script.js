// document.getElementById('pdfUpload').addEventListener('change', function(event) {
//     const file = event.target.files[0];
//     if (file) {
//         console.log('File name:', file.name);
//         // Handle file upload logic here
//     }
// });

document.getElementById("send-text").addEventListener("click", function(event){
    event.preventDefault();
    var text = document.getElementById("input-text").value
    sendTextData(text)
})
function sendTextData(text){
    console.log(JSON.stringify({text}))
    fetch('http://127.0.0.1:8000/summarize_text', {
        method:'POST',
        headers:{
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({text}),
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("text-summary").innerHTML = data['message']
        console.log("Success",data);
    })
    .catch((error) => {
        console.log('Error:', error)
    })
}