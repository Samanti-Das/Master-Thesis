# Implicit Scene Modeling Repo

The readme file only contains instruction on handling the Docker images and containes for this repository. A more detailed documentation for this repository is being written. It is suggested that you use the repository with Docker.  

### Usage with Docker

If you are running on your own computer then you need to first build the Docker image. This requires a CUDA compatible computer with Nvidia drivers already installed. Run the following:  
```
docker build -t imps .
```  
This will take some time.  

It is more likely that you will need Docker images on GPU clusters. There will be an already available Docker image in the GPU cluster named `imps:latest` which will contain the built image of the `master` branch. You should do the following if you would like to create a docker container  


Create a container and connect to it via the following command:
```
docker run -d --gpus "device=<GPU_ID>" -p <PORT>:8888 -v <MOUNT-PATH>:/app/mnt imps:latest
```
- `GPU_ID`: This is the ID of the GPU to utilize. You are only allowed to use one GPU (**if free**) in the GPU server.
- `PORT`: Output port of the Docker server. This will be the port that will be utilized for Jupyter.
- `MOUNT-PATH`: Path in the server/computer to mount to Docker. On the GPU server this will most likely be `/data10`. Inside the container you will be able access the data inside `/data10` via `/app/mnt`.

The Docker container is launched in background. To connect to it `docker attach <CONTAINER_ID>`.  

The suggestion is to utilize Jupyter Lab to edit the code and connect to a remote terminal from your local device. In order to run jupyer lab, First execute the command above, then  

```
jupyter lab --ip 0.0.0.0 --allow-root --no-browser
```  

If you also want to run Tensorboard add another port-forwarding in addition to jupyter's by `-p 6006:6006`. Don't forget to add `--host 0.0.0.0` when you run tensorboard.

Then you will be able to access the jupyter notebook from the port that you have defined while connecting to the docker container.  

**IMPORTANT**
- You will exit the container with `^D`. This will halt and remove the container. Data will be saved but changes in the code will be **LOST**.
- Make sure that you switch to your branch before running jupyter notebook and commit changes before halting the Docker container.
- If you want to run in background, exit the contaienr with `^P^Q`. !NOTE! If you are using the GPU server and nothing is running halt the container with `^D`, other people need the compute power. You can attach back to the container by checking the ID from `docker container ls` and then `docker attach <CONTAINER_ID>`.
- If you want you can also create a build of your extended code in another branch. For that create a docker image with `docker -t imps:<MY_TAG> .`. Nevetheless, this should not be permenent, try to open a 
pull request to the master branch instead of creating your own Docker image. Check the images with `docker image ls` and run the image on a container with your tag instead of `imps:latest`.  

If you are using a host computer with CUDA < 10 it is likely that you will encounter the error:  
```
unsatisfied condition: cuda>=11.0, please update your driver to a newer version, or use an earlier cuda container: unknown.
```

Use the following in `docker run` command to bypass the check done by NVIDIA: `-e NVIDIA_DISABLE_REQUIRE=true`.
