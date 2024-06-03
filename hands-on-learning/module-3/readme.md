## Quickstart

1. Clone the following respository containing the complete code for this module:

    ``
    git clone https://github.com/mage-ai/mlops.git
    ``


2. Change directory into the cloned repo:

    ``
    cd mlops
    ``


3. Pull mageai

   ``
   docker pull mageai/mageai:alpha
   ``
   
4. Run compose

   ``
   docker compose build --no-cache
   ``
   
5. Launch Mage and the database service (PostgreSQL):
   
    ``
    ./scripts/start.sh
    ``

