import ee

path_to_json_file = '/home/stefany/imazon/keys/ee/service-account/sad-impa/dsad-422113-2ecf15fa85de.json'

service_account = 'dsad-impa@dsad-422113.iam.gserviceaccount.com'
credentials = ee.ServiceAccountCredentials(service_account, path_to_json_file)

ee.Initialize(credentials)

print('authenticated')