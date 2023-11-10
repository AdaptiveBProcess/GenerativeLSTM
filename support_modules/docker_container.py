import docker
import tarfile
import os

class DockerContainer:
    def __init__(self, container_image, resources_path, output_path, folder_path):
        self.container_image = container_image
        self.resources_path = resources_path
        self.output_path = output_path
        self.folder_path = folder_path
        self.tar_file_path = output_path + '.tar'

        self.initialize_container()
        self.run_container()
        self.copy_folder_from_container(self.folder_path, self.output_path)
        self.extract_tar_file(self.tar_file_path, self.output_path)
        self.stop_container()

    def test(self):
        exec_command = 'bash run.sh /usr/src/Simod/resources/config.yml outputs/'
        output = self.container.exec_run(exec_command)
        print(output.output.decode())

    def initialize_container(self):
        # Create a Docker client object
        client = docker.from_env()

        # Define the container configuration
        container_config = {
            'image': self.container_image,
            'command': 'bash',
            'volumes': {
                '/path/to/resources/': {'bind': self.resources_path, 'mode': 'rw'},
                '/path/to/output/': {'bind': self.folder_path, 'mode': 'rw'}
            },
            'detach': True,
            'tty': True,
        }

        # Run the Docker container
        self.container = client.containers.run(**container_config)

        # Get the container ID
        self.container_id = self.container.id

        # Check the container status
        print(f"Container status: {self.container.status}")


    def copy_folder_to_container(self, source_folder, destination_folder):
        client = docker.from_env()
        try:
            container = client.containers.get(self.container_id)
            
            # Create a tarball of the source folder
            tar_data = tarfile.open(source_folder + '.tar.gz', 'w:gz')
            tar_data.add(source_folder)
            tar_data.close()
            
            # Read the tarball data
            with open(source_folder + '.tar.gz', 'rb') as file:
                container.put_archive(destination_folder, file.read())
            
            # Remove the temporary tarball
            os.remove(source_folder + '.tar.gz')
            print("Folder copied successfully!")

        except docker.errors.NotFound:
            print("Container not found.")
        except FileNotFoundError:
            print("Source folder not found.")
        except docker.errors.APIError as e:
            print(f"An error occurred: {str(e)}")

    @staticmethod
    def extract_tar_file(tar_file_path, destination_folder):
        try:
            with tarfile.open(tar_file_path, 'r') as tar:
                tar.extractall(destination_folder)
            print("File extracted successfully!")
            os.remove(tar_file_path)
        
        except tarfile.TarError as e:
            print(f"An error occurred: {str(e)}")

    def copy_folder_from_container(self, source_folder, destination_folder):
        client = docker.from_env()
        try:
            container = client.containers.get(self.container_id)
            
            # Retrieve the archive of the source folder from the container
            stream, _ = container.get_archive(source_folder)
            
            # Save the retrieved archive to the destination folder on the local machine
            with open(self.tar_file_path, 'wb') as file:
                for chunk in stream:
                    file.write(chunk)

            
            print("Folder copied successfully!")
        except docker.errors.NotFound:
            print("Container not found.")
        except docker.errors.APIError as e:
            print(f"An error occurred: {str(e)}")

    def delete_folder_in_container(self, folder_path):
        client = docker.from_env()
        try:
            container = client.containers.get(self.container_id)
            command = f"rm -rf {folder_path}"
            container.exec_run(command)
            print("Folder deleted successfully!")
        except docker.errors.NotFound:
            print("Container not found.")
        except docker.errors.APIError as e:
            print(f"An error occurred: {str(e)}")

    def check_folder_exists_in_container(self, folder_path):
        try:
            # Check if the folder exists within the container
            command = f"if [ -d {folder_path} ]; then echo 'exists'; fi"
            result = self.container.exec_run(command, stdout=True, stderr=True)
            
            # Check the command output for folder existence
            output = result.output.decode().strip()
            if output == 'exists':
                self.delete_folder_in_container(folder_path)
            else:
                print("Folder does not exist in the container.")
        except docker.errors.NotFound:
            print("Container not found.")
        except docker.errors.APIError as e:
            print(f"An error occurred: {str(e)}")

    def run_container(self):
        
        # Copy config file into docker container
        source_folder = 'resources'
        destination_folder = '/usr/src/Simod'
        self.copy_folder_to_container(source_folder, destination_folder)

        # Execute a command inside the container
        self.delete_folder_in_container(self.output_path)
        exec_command = f'bash run.sh {self.resources_path}/config.yml outputs/'
        output = self.container.exec_run(exec_command)
        print(output.output.decode())

        # Copy output container folder to proyect folder
        #self.copy_folder_from_container(self.folder_path, self.output_path)

    def stop_container(self):
        
        # Stop and remove the container
        self.container.stop()
        self.container.remove()
