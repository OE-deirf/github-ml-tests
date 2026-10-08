# github-docker-mlflow
Simple Machine Learning demo using docker in github + MLFlow

# set settings
- .env file <- .env.example file
- secrets directory <- secrets.example directory

# venv
- python3.10 -m venv .venv
- source .venv/bin/activate
- pip install -v -r requirements.txt/requirements_dev.txt

- uv venv --python 3.10
- source .venv/bin/activate
- uv pip install -r requirements.txt/requirements_dev.txt

# DVC, GIT  init
- git init
- dvc init            # create .dvc/ dir
- git commit -m "DVC init"

# DVC config
- dvc config core.autostage true        # after every dvc auto git add as well
- dvc config cache.type reflink,hardlink,symlink,copy   # cache -> workspace link type
- dvc config --list                      # actual config

# DVC remote
- dvc remote add -d storage s3://bucket/projekt     # -d = default remote
- dvc remote add backup /mnt/nas/dvc                # local / remote dir
- dvc remote add ssh_r ssh://user@host/path

# avoid secret to git:
- dvc remote modify --local storage access_key_id XXX
- dvc remote modify --local storage secret_access_key YYY

- dvc push        # cache -> remote
- dvc fetch       # remote-> cache
- dvc pull        # fetch + checkout

# DVC commit
- dvc add data/raw                    # -> data/raw.dvc + data/.gitignore
- git add data/raw.dvc data/.gitignore
- git commit -m "Data v1"
- dvc push

# DVC after modified data:
- dvc status
- dvc commit data/raw.dvc  # or just dvc add data/raw
- git commit -am "Data v2"
- dvc push

# checkout
- git pull
- git checkout v1.0
- dvc pull
- dvc checkout

# checkout only with old data version
- git checkout v1.0 -- data/raw.dvc
- dvc checkout data/raw.dvc

# dvg dag
- dvc repro          # rerun the changed stage
- git add dvc.yaml dvc.lock
- git commit -m "Pipeline"
- dvc push
- dvc dag            # print DAG
  - dvc dag --outs   # just files
  - dvc dag train    # just train and its anchients
  - dvc dag --dot | dot -Tpng -o dag.png    # Graphviz
  - dvc dag --mermaid / --md                # Mermaid diagram

# full example
# local
- vim params.yaml
- dvc repro
- dvc metrics diff
- git commit -am "lr=0.01"
- dvc push

# other local
- git pull
- dvc pull

# dvc system status
- dvc doctor

# docker build
- docker build --network=host -t <image_name>:<tag> .
# docker run
- docker run -d --rm --name <name>
- docker run --name <name> <image_name>:<tag>
# check logs
- docker logs -f <container_id>

# compose.yaml example
services:
  api:
    build:
      context: ./api           # build context
      dockerfile: Dockerfile   # based on the context
      target: production       # stage in case of multi-stage build
      args:
        PYTHON_VERSION: "3.12" # Dockerfile ARGs
    image: myorg/api:latest    # builded image name (optional)
    ports:
      - "8000:8000"

  db:
    image: postgres:16         # not build just pull

# docker compose
- docker compose config
- docker compose build                          # every buildable service
- docker compose build <service_name>           # only one service
- docker compose build --no-cache               # without cache, full rebuild
- docker compose build --pull                   # pull the FROM base image
- docker compose up --build                     # build, and run
- docker compose up -d --build <service_name>   # only one service rebuild and restart
- docker compose down -v --remove-orphans       # remove all containers and data
- docker compose exec <service_name> bash       # run a service with bash command
- docker compose ps --services                  # prit all running containers

# docker container
- docker container list -a                       # list all containers
- docker container rm <container id>..           # delete a container

# docker image
- docker image list -a                           # list all images
- docker image rm <image name>..                 # delete an image
