import earthaccess

print("Loging in:")
auth = earthaccess.login(strategy="interactive", persist=True)

print("Getting AST_08 Granules:")
granules = earthaccess.search_data(
    short_name="AST_08",
    version="004",
    temporal=("1991-01-01","2025-12-31"),
    polygon=[[-67.897909, -21.740668], [-67.685208,-21.740668], [-67.685208, -21.567356], [-68.220342, -21.567356], [-67.897909, -21.740668]],
    day_night_flag="NIGHT"
)

print("Downloading Granules:")

earthaccess.download(granules, "./AST08/raw")
