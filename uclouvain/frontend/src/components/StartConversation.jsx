import { Box, Button, Card, CardContent, CardHeader, IconButton, Input, TextField, Typography } from "@mui/material";
import SendIcon from '@mui/icons-material/Send';
import classes from "./StartConversation.module.css"

export default function StartConversation({title, question, set_question}) {
    return (<>
            <Box textAlign="center">
                <img src="logo.png" />
            </Box>
            <Box sx={{padding: "1em"}}>
                <Typography variant="h6" textAlign="center">{title}</Typography>
            </Box>
            <input htmltype="text" className={classes.TextInput} onInput={(txt) => {set_question(txt.target.value)}} />
            <Box textAlign="right" marginRight="10px" marginTop="-45px">
                <IconButton color="primary" onClick={() => {console.log(text)}}><SendIcon /></IconButton>
            </Box>
    </>);
}