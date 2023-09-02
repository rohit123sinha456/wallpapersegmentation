### Methods to deploy this code
- git clone this repository
- cd intp api
- run ```docker build -t app .```
- run ```docker run -d -p 5000:5000 api```

## If there is any issue with docker or memory run with WSGI app
- Install python in the instance
- git clone this repository
- cd into the directory
- run ```python -m venv env```
- Activate the virtual env ``` source env/bin/activate```
- Run ```pip install -r requirements.txt```
- cd into the api  directory ```cd api```
- Run ```gunicorn -w 3 -b :5000 -t 0 --reload --daemon wsgi:app```


## To Test
- After running the app
- open the wallsegment.postman_collection.json file in POSTMAN app
- Use the POST Request to get the infered image name from the JSON returned
- Use the GET request to get the image (The image name from the revious step must be passed on as the query parameter)


