document.getElementById('emailForm').addEventListener('submit', async function (event) {
    event.preventDefault();
    const emailText = document.getElementById('emailText').value;
    if(emailText == '' || emailText== undefined)
    {
        validateEmail()
        return false;
    }
    const response = await fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text: emailText })
    });
    const result = await response.json();
    result.spam ? spam() : notspam();
});

function spam(){
    Swal.fire({
        title: "Spam",
        text: "Based on the analysis, this email appears to be classified as spam.",
        icon: "warning"
      });
}

function notspam()
{
    Swal.fire({
        title: "Not Spam",
        text: "Based on the analysis, this email appears to be safe and not classified as spam.",
        icon: "success"
      });
}
function validateEmail()
{
    Swal.fire({
        title: "Error",
        text: "Please enter email text",
        icon: "error"
      });
}
function ClearText(){
    const emailTextElement = document.getElementById('emailText');
    emailTextElement.value = '';
}
{/* <script>
    async function classifyEmail() {
        const message = document.getElementById('message').value;
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message }),
        });
        const data = await response.json();
        const resultDiv = document.getElementById('result');
        if (data.spam) {
            resultDiv.innerHTML = `Prediction: <strong>Spam</strong><br>Confidence: ${(data.confidence * 100).toFixed(2)}%`;
            resultDiv.style.color = 'red';
        } else {
            resultDiv.innerHTML = `Prediction: <strong>Not Spam</strong><br>Confidence: ${(data.confidence * 100).toFixed(2)}%`;
            resultDiv.style.color = 'green';
        }
    }
</script> */}