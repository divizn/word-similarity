.PHONY: help up down logs redis-cli build run clean

help: # Show help for each of the Makefile recipes.
	@grep -E '^[a-zA-Z0-9 -]+:.*#'  Makefile | sort | while read -r l; do printf "\033[1;32m$$(echo $$l | cut -f 1 -d':')\033[00m:$$(echo $$l | cut -f 2- -d'#')\n"; done

up: # Start Redis container in the background
	docker-compose up -d  # Launch Redis as background container

down: # Stop Redis container
	docker-compose down  # Stop and remove Redis container

logs: # Tail Redis container logs
	docker-compose logs -f  # View Redis logs live

redis-cli: # Open interactive Redis CLI
	docker exec -it redis-embeddings redis-cli  # Connect to Redis CLI inside container

build: # Compile the Rust project
	cargo build --release  # Build Rust project in release mode

run: # Run the Rust project, starts Redis if it is not running
	@echo "Checking if Redis container is running..."
	@if [ $$(docker ps -q -f name=redis-embeddings) ]; then \
		echo "Redis is already running."; \
	else \
		echo "Redis not running. Starting container..."; \
		docker compose up -d; \
	fi
	cargo run --release  # Execute Rust program using release build

clean: # Remove build artifacts
	cargo clean  # Clean Rust target directory
