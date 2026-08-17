CREATE TABLE Jobs (PRIMARY KEY(Job_Code)) as

SELECT
	DISTINCT Job_code,
    Job_Title
    
FROM
	company_data.company_employees;