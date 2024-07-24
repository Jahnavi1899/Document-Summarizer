document.getElementById('pdfUpload').addEventListener('change', function(event) {
    const file = event.target.files[0];
    console.log(file)
    if (file) {
        var filename = file.name
        console.log('File name:', filename);
        // Handle file upload logic here
    }

});

document.getElementById('file-upload').addEventListener("click", function(event){
    event.preventDefault()
    const fileInput = document.getElementById('pdfUpload')
    const file = fileInput.files[0]
    if(file){
        uploadFile(file)
    }
    else{
        alert("Please upload a file!")
    }
})

document.getElementById("send-text").addEventListener("click", function(event){
    event.preventDefault();
    var text = document.getElementById("input-text").value
    sendTextData(text)
})

function uploadFile(file){
    console.log("Inside uploadFile method")
    const formData = new FormData();
        formData.append('file', file);

        fetch('http://127.0.0.1:8000/upload_file', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            response.json()
        })
        .then(data => {
            console.log("PDF Upload Success", data);
        })
        .catch((error) => {
            console.log('Error:', error);
    });
}

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

