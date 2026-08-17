DROP TABLE IF EXISTS Employees;

CREATE TABLE Employees ( PRIMARY KEY(Employee_id)) as
SELECT
	DISTINCT Employee_id,
    Name,
    State_code,
    Home_state
    
FROM
	company_data.company_employees;