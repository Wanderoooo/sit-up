import requests
from datetime import datetime

domain = "hanson"
app_id = "1"
api_token = "SnWU4qGebfkdUbpRD9oB4XvUCZ9xesmEKda1Hyti"

post_url = f"https://{domain}.kintone.com/k/v1/record.json"
url = f"https://{domain}.kintone.com/k/v1/records.json?app={app_id}"

request_headers = {
  "X-Cybozu-API-Token": api_token,
}

send_headers = {
    "X-Cybozu-API-Token": api_token,
    "Content-Type": "application/json"
}

def parse_record(record):
    date = datetime.strptime(record["Date_0"]["value"], "%Y-%m-%d").date()
    modelData = record["Text"]["value"] # some how need to convert this to binary
    startTime = datetime.strptime(record["Time_1"]["value"], "%H:%M").time()
    endTime = datetime.strptime(record["Time_2"]["value"], "%H:%M").time()
    percentage = float(record["Number_1"]["value"])
    length = float(record["Number_2"]["value"])

    return {"date": date, "modelData": modelData, "startTime": startTime, "endTime": endTime, "percentage": percentage, "length": length}

def get_records():
    response = requests.get(url, headers=request_headers)
    print("jello??")
    if response.status_code == 200:
        data = response.json()
        records = data["records"]
        print("please??")
        return [parse_record(record) for record in records]
    
    print(f"Error: {response.status_code}")
    return []

def add_record_record(record):
    data = {
        "app": app_id,
        "record": {
            "Date_0": {"value": record["date"].strftime("%Y-%m-%d")},
            "Text":   {"value": record["modelData"]},
            "Time_1": {"value": record["startTime"].strftime("%H:%M")},
            "Time_2": {"value": record["endTime"].strftime("%H:%M")},
            "Number_1": {"value": str(record["percentage"])}, 
            "Number_2": {"value": str(record["length"])}
        }
    }

    response = requests.post(post_url, headers=request_headers, json=data)
    if response.status_code == 200:
        return True
    
    print(f"Error: {response.status_code}")
    return False

def add_record(date, model, start, end, percentage):
    data = {
        "app": app_id,
        "record": {
            "Date_0": {"value": date.strftime("%Y-%m-%d")},
            "Text": {"value": model},
            "Time_1": {"value": start.strftime("%H:%M")},
            "Time_2": {"value": end.strftime("%H:%M")},
            "Number_1": {"value": str(percentage)}, 
            "Number_2": {"value": str((end - start).total_seconds())}
        }
    }

    response = requests.post(post_url, headers=send_headers, json=data)
    if response.status_code == 200:
        return True
    
    print(f"Error: {response.status_code}")
    print(response.text)
    return False

# if response.status_code == 200:
#     data = response.json()
#     records = data["records"]
#     for record in records:
#         print(record)
#         date = datetime.strptime(record["Date_0"]["value"], "%Y-%m-%d").date()
#         modelData = record["Text"]["value"] # some how need to convert this to binary
#         startTime = datetime.strptime(record["Time_1"]["value"], "%H:%M").time()
#         endTime = datetime.strptime(record["Time_2"]["value"], "%H:%M").time()
#         percentage = float(record["Number_1"]["value"])
#         length = float(record["Number_2"]["value"])

#         print(f"Date: {date}, Model Data: {modelData}, Start Time: {startTime}, End Time: {endTime}, Percentage: {percentage}, Length: {length}")

if __name__ == "__main__":

    add_record(datetime.now(), "model", datetime.now(), datetime.now(), 0.5)

    records = get_records()
    for record in records:
        print(record)
    pass