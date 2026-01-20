fetch('http://127.0.0.1:5000/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ features: [1.2, 3.4, 5.6, 7.8, 9.0, 2.1, 4.3, 5.9, 6.7, 8.4]})
})
.then(response => {
    if (!response.ok) {
        return response.json().then(err => Promise.reject(err));
    }
    return response.json();
})
.then(data => console.log('Prediction:', data.prediction))
.catch(error => console.error('Error:', error));
