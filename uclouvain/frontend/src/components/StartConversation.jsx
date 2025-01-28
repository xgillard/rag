import { Box, Button, Card, CardContent, CardHeader, IconButton, Input, TextField, Typography } from "@mui/material";
import SendIcon from '@mui/icons-material/Send';
import classes from "./StartConversation.module.css"

export default function StartConversation({title, set_question, send_action}) {
    function on_keydown(event) {
        if (event.keycode == 13) {
            send_action();
        }
    }
    function on_input(event) {
        set_question(event.target.value);
    }
    return (<>
            <Box textAlign="center">
                <img src="logo.png" />
            </Box>
            <Box sx={{padding: "1em"}}>
                <Typography variant="h6" textAlign="center">{title}</Typography>
            </Box>
            <textarea className={classes.TextInput} onInput={on_input} onKeyDown={on_keydown} />
            <Box textAlign="right" marginRight="10px" marginTop="-45px">
                <IconButton color="primary" onClick={send_action}><SendIcon /></IconButton>
            </Box>
    </>);
}