import requests

url = "https://16c73e6f9eec.ngrok-free.app/api/v1/hackrx/run"
headers = {
    "Authorization": "Bearer Token Here",
    "Content-Type": "application/json"
}
data = {
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]
}

response = requests.post(url, headers=headers, json=data)

# Print response details for debugging
print(f"Status Code: {response.status_code}")
print(f"Response Headers: {dict(response.headers)}")
print(f"Response Content: {response.text}")
print(f"Response Content Type: {response.headers.get('content-type', 'Not specified')}")

# Only try to parse JSON if we have content
if response.text.strip():
    try:
        json_response = response.json()
        print("JSON Response:", json_response)
    except requests.exceptions.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        print("Raw response is not valid JSON")
else:
    print("Response is empty")