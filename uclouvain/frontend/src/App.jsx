import { useCallback, useState } from 'react';
import { Backdrop, CircularProgress, Container, CssBaseline, Paper } from '@mui/material';
import { MuiMarkdown } from 'mui-markdown';
import StartConversation from './components/StartConversation';
import Answer from './components/Answer';

function App() {
  const title = "Que puis-je faire pour vous aujourd'hui ?";

  const [question, set_question] = useState();
  const [waiting,  set_waiting]  = useState(false);
  const [response, set_response] = useState();

  const send_action = useCallback(async function() {
    set_waiting(true);

    const response = await fetch(`${import.meta.env.VITE_API_URL}/rag`,
      {
        method:  "POST",
        headers: {
          "Content-Type": "application/json; charset=utf-8",
          "Access-Control-Allow-Origin": "*",
        },
        body: JSON.stringify({
          "question": question,
          "search_results": 5,
          "generation_tokens": 256,
          "append_sources": false
        })
      }
    );
    
    if (!response.ok) {
      console.log(response)
      throw Error("Bad response");
    } else {
      const reader  = response.body.getReader();
      const decoder = new TextDecoder();
      while(true) {
        const {done, value} = await reader.read();
        if (done) {
          break;
        }
        set_waiting(false);
        set_response((r) => {
          const chunk = decoder.decode(value, {stream: true});
          return r ? r + chunk : chunk;
        });
      }
    }

  }, [question, set_response])

  return (
      <>
        <CssBaseline />
        <Container maxWidth="lg" sx={{alignContent: "center", paddingTop: "5em"}}>
          {(response == null && <StartConversation title={title} set_question={set_question} send_action={send_action} />)}
          {(waiting          && <Backdrop open={waiting} ><CircularProgress /></Backdrop>)}
          {(response         && <Answer question={question} response={response} />)}
        </Container>
      </>
  )
}

export default App
