SELECT table_name 
FROM information_schema.tables
WHERE table_schema = 'marketing_roi_analysis';

SELECT table_name, table_rows
FROM information_schema.tables
WHERE table_schema = 'marketing_roi_analysis';