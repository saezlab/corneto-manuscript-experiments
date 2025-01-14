@echo off
REM Define the arrays for alpha and num_nodes
set alphas=0.05 0.10 0.15 0.20 0.25 0.30
set nodes=250 500 1000

REM Run the Python script with each combination of alpha and num_nodes
for %%a in (%alphas%) do (
    for %%n in (%nodes%) do (
        echo Running Python script with alpha=%%a and num_nodes=%%n
        python gen_steiner_dataset.py --strict_acyclic --num_samples 10 --alpha %%a --num_nodes %%n
        if errorlevel 1 (
            echo Error executing Python script with alpha=%%a and num_nodes=%%n
            pause
            exit /b 1
        )
    )
)

echo Script execution completed. Press any key to exit.
pause
