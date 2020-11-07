echo -e "\e[1m>>> Running checks for roe.com"
echo -e "\e[0m"

# Check for virtualenv
echo -ne "\e[1m>> Checking if 'virtualenv' is present: \e[0m"
venv_bool=$(pip3 freeze | grep virtualenv)

if [[ $venv_bool == "" ]]
then
	echo -e "No.\nStarting Installation:"
	pip3 install virtualenv
else
	echo -e "Yes."
fi

# Check for virtual environment
echo -ne "\e[1m>> Checking if virtual environment is present: \e[0m"
if [[ -d "./env" ]]
then
        echo -e "Yes."
else
        echo -e "No.\nCreating Virtual Environemt:"
        virtualenv env
fi

if [[ -f "requirements.txt" ]]
then
	source "env/bin/activate"
	pip install -r requirements.txt
fi