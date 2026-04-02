
import msal
import requests
import base64

# Your sharing link
sharing_url = ("https://prodaptcloud-my.sharepoint.com/:x:/g/personal/dinesh_sj_prodapt_com/"
               "IQCQAyzvL-pdQZEY6vmxPapOAdyoMQjqxJxYwXVA-81LrC0?e=VODpIq")

# Encode link for Graph shares API
b64 = base64.b64encode(sharing_url.encode("utf-8")).decode("ascii")
encoded = "u!" + b64.rstrip("=").replace("/", "_").replace("+", "-")

# MSAL Public Client (Device Code Flow)
CLIENT_ID = "d3590ed6-52b3-4102-aeff-aad2292ab01c"  # Use Microsoft’s default or register a simple app
AUTHORITY = "https://login.microsoftonline.com/common"
SCOPES = ["Files.Read"]  # Delegated permission

app = msal.PublicClientApplication(CLIENT_ID, authority=AUTHORITY)
flow = app.initiate_device_flow(scopes=SCOPES)
if "user_code" not in flow:
    raise RuntimeError("Failed to create device flow")

print(f"Go to {flow['verification_uri']} and enter code: {flow['user_code']}")
result = app.acquire_token_by_device_flow(flow)
if "access_token" not in result:
    raise RuntimeError(result.get("error_description", "Auth failed"))

token = result["access_token"]
headers = {"Authorization": f"Bearer {token}"}

# Resolve to DriveItem
item_resp = requests.get(f"https://graph.microsoft.com/v1.0/shares/{encoded}/driveItem", headers=headers)
item_resp.raise_for_status()
item = item_resp.json()
drive_id = item["parentReference"]["driveId"]
item_id = item["id"]

# Download file
content_url = f"https://graph.microsoft.com/v1.0/drives/{drive_id}/items/{item_id}/content"
file_resp = requests.get(content_url, headers=headers)
file_resp.raise_for_status()

with open("downloaded.xlsx", "wb") as f:
    f.write(file_resp.content)

print("File saved as downloaded.xlsx")