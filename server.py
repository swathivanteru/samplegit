from fastmcp import FastMCP
from pydantic import BaseModel
from pymongo import MongoClient

# Initialize MCP App
app = FastMCP("Device Management Server")

# MongoDB Connection
client = MongoClient("mongodb://localhost:27017/")
db = client["Anomaly"]
collection = db["Network_anomaly"]

# Request Models
class FetchRequest(BaseModel):
    limit: int = 100  

class UpdateRequest(BaseModel):
    device_id: str
    updates: dict  # Dictionary of fields to update

class RestartRequest(BaseModel):
    device_id: str
    device_behavior_score: float

# Tools
@app.tool(description="Fetch device data from MongoDB")
def fetch_devices(request: FetchRequest):
    try:
        data = list(collection.find({}, {"_id": 0}).limit(request.limit))
        return {"devices": data}
    except Exception as e:
        return {"error": str(e)}

@app.tool(description="Update device status in MongoDB")
def update_device(request: UpdateRequest):
    collection.update_one({"device_id": request.device_id}, {"$set": request.updates})
    return {
        "status": "updated",
        "device_id": request.device_id,
        "updates": request.updates
    }

@app.tool(description="Restart a device based on its behavior score")
def restart_device(request: RestartRequest):
    print(f"🔁 Restarting device {request.device_id} with score {request.device_behavior_score}")
    return {
        "device_id": request.device_id,
        "restarted": True,
        "score": request.device_behavior_score
    }

# Run MCP Server
if __name__ == "__main__":
    app.run(transport="sse")