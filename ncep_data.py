import requests

# Define your request parameters
base_url = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl"
params = {
    "file": "gfs.t00z.pgrb2.0p25.f006",  # 6-hour forecast
    "lev_1000_mb": "on",                 # Level
    "var_PRMSL": "on",                   # Variable (MSLP)
    "subregion": "",                     # Use subregion to specify bounding box
    "leftlon": -130,
    "rightlon": -60,
    "toplat": 50,
    "bottomlat": 20,
    "dir": "/gfs.20221001/00"            # Date and time (forecast)
}

# Prepare the request
req = requests.Request('GET', base_url, params=params)
prepared = req.prepare()

# Print the prepared request URL
print(prepared.url)

# Send the request using a session
with requests.Session() as session:
    response = session.send(prepared)


# Save the response content as a file if the request is successful
if response.status_code == 200:
    with open("subset_mslp_6h.grb", "wb") as f:
        f.write(response.content)
    print("Subset download successful.")
else:
    print(response.content)
    print(f"Error: {response.status_code}")
