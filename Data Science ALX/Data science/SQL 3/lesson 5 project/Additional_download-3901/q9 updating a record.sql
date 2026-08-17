-- Start a new transaction
START TRANSACTION;

-- Execute the UPDATE statement
UPDATE md_water_services.employee
SET phone_number = '+99643864786'
WHERE employee_name = 'Bello Azibo';

-- Rollback the transaction to undo changes
ROLLBACK;
