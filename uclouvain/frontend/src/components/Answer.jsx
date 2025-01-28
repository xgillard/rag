import { Paper, Typography } from "@mui/material";
import MuiMarkdown from "mui-markdown";

import classes from "./Answer.module.css";

export default function Answer({question, response}) {
    return (
        <>
        <div className={classes.Question}>
            <Typography>{question}</Typography>
        </div>
        <div className={classes.Answer}>
            <MuiMarkdown>{response}</MuiMarkdown>
        </div>
        </>
    );
}