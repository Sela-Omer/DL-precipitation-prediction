from ecmwfapi import ECMWFDataServer
import time

api_key = {
    "url": "https://api.ecmwf.int/v1",
    "key": "3ad64386f6aa650387182b5700a58475",
    "email": "omer.sela13@gmail.com"
}
server = ECMWFDataServer(**api_key)


# Example Unix timestamp (can be set dynamically)
forecast_time_unix = 1704070800

# Convert Unix timestamp to ECMWF date format (e.g., '2023-01-01')
forecast_date = time.strftime('%Y-%m-%d', time.gmtime(forecast_time_unix))

# Specify your area (format: North/West/South/East in lat/lon)
area_variable = '50/-10/30/40'  # Replace with your desired area

server.retrieve({
    'class': 'od',            # Operational data
    'stream': 'oper',          # Operational forecast stream
    'datasets': 'archive',       # ERA-Interim dataset
    'levtype': 'sfc',          # Surface level data
    'param': '151.128',  # MSL parameter for Mean Sea Level Pressure
    'date': forecast_date,      # Forecast date: 6th March 1988
    'time': '00:00:00',        # Forecast time: 12 UTC
    'step': '6',               # Forecast step (0 for analysis)
    'type': 'fc',              # Forecast type
    'expver': '1',             # Experiment version 1
    'target': 'output.grib',   # File name for the output (GRIB format)
    'grid': '0.25/0.25',       # Optional: resolution in degrees (e.g., 0.25 degree grid)
    'area': area_variable,  # Variable for the geographical area
    'format': 'netcdf',  # Output format
})
