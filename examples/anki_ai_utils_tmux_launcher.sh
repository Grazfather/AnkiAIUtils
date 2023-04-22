#!/usr/bin/zsh
aiutilpath="PATH TO ANKI AI UTILS"
tmux_mode=0  # wether to launch using tmux or not
START_DELAY=10
WAIT_TIME_FOR_INTERNET=30  # if internet seems unresponsive, wait that much
CMD_SEP=";"  # separator between the different tools
PDB=""  # wether to launch python with pdb or not
DEBUG=""  # wether to add --debug to the tool call
FORCE=""  # wether to add --force or not
EXIT="&& exit"  # wether to exit after the runs or not

notif_arg="--ntfy_url ntfy.sh/$NTFY_PHONE"
common_query='deck:yourdeck -deck:not::this::deck note:Clozolkor -tag:not::that::tag -is:suspended'
query="rated:2:1 OR rated:2:2"  # failed or hard of the last 2 days
anchors_file="./anchors.json"  # your examples

# gather user arguments
while (( $# > 0 )); do
    case "$1" in
        -t | --tmux_mode)
            tmux_mode="$2"
            echo "Set tmux_mode to $tmux_mode"
            shift 2
            ;;
        -q | --query)
            query="$2"
            echo "Set query to $query"
            shift 2
            ;;
        --fail)
            CMD_SEP="&&"
            echo "Will crash if one command fails, otherwise keep going."
            shift 1
            ;;
        --pdb)
            PDB="-m pdb"
            shift 1
            echo "Will use pdb to launch python"
            ;;
        --now)
            START_DELAY=0
            WAIT_TIME_FOR_INTERNET=0
            echo "Starting now"
            shift 1
            ;;
        --debug)
            DEBUG="--debug"
            echo "Using arg --debug for each util"
            shift 1
            ;;
        --force)
            FORCE="--force"
            echo "Using arg --force for each util"
            shift 1
            ;;
        --noexit)
            EXIT=""
            echo "Script will not exit the session at the end (the default is to exit if no error occured)"
            shift 1
            ;;
        --no_deck_restriction)
            common_query='note:Clozolkor -tag:pro::Externat::particularite_dankifiage -is:suspended'
            echo "Will not apply deck restriction"
            echo "Commong query: $common_query"
            shift 1
            ;;
        *)
            echo "Invalid option(s): $@"
            exit 1
            ;;
    esac
done



# send notification to phone
function phone_notif() {
    sender=$NTFY_PHONE
    title="AnkiExpIllusMem"
    message="$1"
    curl -s -H "Title: $title" -d "$message" "ntfy.sh/$sender"
}

# echo and send notification to user phone
function logg() {
    phone_notif "$1"
    echo "$1"
}

# exit if session already running
if [[ $tmux_mode -eq 1 ]]
then
    base_session="LOCAL_Anki_AI_Utils"
    n_sess=$(tmux list-sessions |grep $base_session | wc -l)
    if [ ! $n_sess -eq 0 ]
    then
        logg "Error, session already running?"
        exit 1
    fi
fi


# Loop until internet connection is found
while ! ping -c 1 google.com &> /dev/null
do
  logg "No internet connection found, waiting ${WAIT_TIME_FOR_INTERNET} seconds..."
  sleep ${WAIT_TIME_FOR_INTERNET}
done


#sync anki
logg "syncing anki"
curl -s localhost:8765 -X POST -d '{"action": "sync", "version": 6}'
sleep 5

# exit if contains a deck called "nosync"
deck_status=$(curl -s localhost:8765 -X POST -d '{"action": "deckNames", "version": 6}' | jq | tr '[:upper:]' '[:lower:]' | grep "nosync" | wc -l)
if [ ! $deck_status -eq 0 ]
then
    logg "Error, contains deck 'nosync'?"
    exit 1
fi


session_id=`date +%d.%m.%Y_%Hh%M_%S`
session_name=$(echo "$base_session"_"$session_id")

if [[ $tmux_mode -eq 1 ]]
then
    if [ ! -z "$TMUX" ]
    then
        # rename session if already in tmux
        tmux rename-session $session_name
    else
        # or create tmux session
        session=$session_name
        window=${session}:0
        pane=${window}.0
        tmux new-session -d -s $session_name
    fi
fi



# warn user
logg "Starting in $START_DELAY second"
sleep $START_DELAY

sum="(cd $aiutilpath && python $PDB explainer.py \
    --field_names body \
    $DEBUG \
    $FORCE \
    --query \"($query OR tag:AnkiIllustrator::todo OR tag:AnkiIllustrator::failed) -tag:AnkiExplainer::to_keep $common_query\" \
    --dataset_path author_dir/explainer_dataset.txt \
    --string_formatting author_dir/string_formatting.py \
    $notif_arg || phone_notif 'Anki_AI_Utils' 'error summary' )"
mnem="(cd $aiutilpath && python $PDB mnemonics.py \
    --field_names body \
    $DEBUG \
    $FORCE \
    --query \"($query OR tag:AnkiIllustrator::todo OR tag:AnkiIllustrator::failed) -tag:AnkiMnemonics::to_keep $common_query\" \
    --memory_anchors_file $anchors_file \
    --dataset_path author_dir/mnemonics_dataset.txt \
    --string_formatting author_dir/string_formatting.py \
    $notif_arg || phone_notif 'Anki_AI_Utils' 'error mnemonics')"
illus="(cd $aiutilpath && python $PDB illustrator.py \
    --field_names body \
    $DEBUG \
    $FORCE \
    --query \"($query OR tag:AnkiIllustrator::todo OR tag:AnkiIllustrator::failed) -tag:AnkiIllustrator::to_keep -body:*img* $common_query\" \
    --memory_anchors_file $anchors_file \
    --dataset_path author_dir/illustrator_dataset.txt \
    --dataset_sanitize_path author_dir/illustrator_sanitize_dataset.txt \
    --string_formatting author_dir/string_formatting.py \
    $notif_arg || phone_notif 'Anki_AI_Utils' 'error illustrator')"
cmd=$(echo "$sum $CMD_SEP $mnem $CMD_SEP $illus $CMD_SEP $EXIT")

echo "\n\n"
echo "Command to execute:\n$cmd"
echo "\n\n"


if [[ $tmux_mode -eq 1 ]]
then
    tmux send-keys -- $cmd Enter
    logg "Finished launch script!"
else
    logg "Executing launch script!"
    eval $cmd
fi



