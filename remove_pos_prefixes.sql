-- Script to remove POS prefixes from current_line in the definitions table
-- This removes patterns like "_noun_ ", "_verb_ ", "_adjective_ ", etc. from the beginning of current_line

-- First, let's see what we're working with
SELECT word, current_line
FROM definitions
WHERE current_line LIKE '\_%\_ %'
LIMIT 10;

-- Update the current_line to remove POS prefixes
-- Pattern matches: _<pos>_ at the beginning followed by a space
UPDATE definitions
SET current_line = SUBSTR(current_line, INSTR(current_line, ' ') + 1)
WHERE current_line LIKE '\_%\_ %'
  AND INSTR(current_line, ' ') > 0;

-- Verify the changes
SELECT word, current_line
FROM definitions
WHERE current_line LIKE '\_%\_ %'
LIMIT 10;